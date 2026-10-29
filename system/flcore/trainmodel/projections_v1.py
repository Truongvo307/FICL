"""
Bidirectional Projection Module for RiemannFL (Idea 1)

Contains:
- Forward projection: Φ_i: R^{d_i} → R^d
- Inverse projection: Φ_i⁻¹: R^d → R^{d_i}

Both are learned during initialization phase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class BidirectionalProjection(nn.Module):
    """
    Bidirectional projection for RiemannFL.
    
    Forward:  Φ_i: R^{d_i} → R^d  (encode to shared space)
    Inverse:  Φ_i⁻¹: R^d → R^{d_i}  (decode back to client space)
    
    Both are learned networks, not just pseudo-inverse.
    """
    
    def __init__(
        self,
        client_dim: int,      # d_i: client-specific representation dimension
        shared_dim: int,      # d: shared latent dimension
        hidden_dims: Optional[List[int]] = None,
        normalize: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.client_dim = client_dim
        self.shared_dim = shared_dim
        self.normalize = normalize
        
        # Build forward projection: R^{d_i} → R^d
        self.encoder = self._build_network(
            input_dim=client_dim,
            output_dim=shared_dim,
            hidden_dims=hidden_dims,
            activation=activation
        )
        
        # Build inverse projection: R^d → R^{d_i}
        # Use reversed hidden dims for symmetry
        inverse_hidden = list(reversed(hidden_dims)) if hidden_dims else None
        self.decoder = self._build_network(
            input_dim=shared_dim,
            output_dim=client_dim,
            hidden_dims=inverse_hidden,
            activation=activation
        )
        
        self._init_weights()
    
    def _build_network(
        self, 
        input_dim: int, 
        output_dim: int,
        hidden_dims: Optional[List[int]],
        activation: str
    ) -> nn.Module:
        """Build a feedforward network."""
        layers = []
        current_dim = input_dim
        
        if hidden_dims:
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(self._get_activation(activation))
                current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_project(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward projection: Φ_i(z) 
        
        Args:
            z: [batch, d_i] or [d_i] - client-specific representation
        
        Returns:
            z_tilde: [batch, d] or [d] - shared space representation
        """
        z_tilde = self.encoder(z)
        
        if self.normalize:
            z_tilde = F.normalize(z_tilde, p=2, dim=-1)
        
        return z_tilde
    
    def inverse_project(self, w: torch.Tensor) -> torch.Tensor:
        """
        Inverse projection: Φ_i⁻¹(w)
        
        Args:
            w: [batch, d] or [d] - shared space representation (e.g., barycenter)
        
        Returns:
            w_hat: [batch, d_i] or [d_i] - client-specific representation
        """
        return self.decoder(w)
    
    def forward(self, z: torch.Tensor) -> tuple:
        """
        Full forward pass: encode then decode (for reconstruction loss).
        
        Returns:
            z_tilde: projected representation
            z_rec: reconstructed representation
        """
        z_tilde = self.forward_project(z)
        z_rec = self.inverse_project(z_tilde)
        return z_tilde, z_rec


class ProjectionInitializer:
    """
    Learns bidirectional projections for all clients using public data.
    
    This is Phase 0 of RiemannFL - runs ONCE before federated training.
    
    Loss = λ_recon * L_reconstruction + λ_cons * L_consistency
    
    Where:
    - L_reconstruction = ||z - Φ_i⁻¹(Φ_i(z))||²  (preserve information)
    - L_consistency = KL(Φ_i(z) || Φ_j(z))  (align across clients)
    """
    
    def __init__(
        self,
        clients: list,
        shared_dim: int,
        device: torch.device,
        public_data_loader,
        init_epochs: int = 20,
        init_lr: float = 0.01,
        lambda_recon: float = 1.0,
        lambda_cons: float = 0.5,
        hidden_dims: Optional[List[int]] = None
    ):
        self.clients = clients
        self.shared_dim = shared_dim
        self.device = device
        self.public_loader = public_data_loader
        self.init_epochs = init_epochs
        self.init_lr = init_lr
        self.lambda_recon = lambda_recon
        self.lambda_cons = lambda_cons
        self.hidden_dims = hidden_dims or [256]
        
        self.projections = {}  # client_id -> BidirectionalProjection
    
    def _get_client_rep_dim(self, client, sample_batch) -> int:
        """Determine representation dimension for a client's model."""
        from flcore.clients.clientbase import load_item
        
        model = load_item(client.role, 'model', client.save_folder_name)
        if model is None:
            return 512
        
        model.eval()
        with torch.no_grad():
            x = sample_batch[0][:2].to(self.device)
            rep = model.base(x)
            if rep.dim() == 4:
                rep = rep.view(rep.size(0), -1)
            elif rep.dim() == 3:
                rep = rep.mean(dim=1)
            return rep.shape[-1]
    
    def initialize(self):
        """Create projection modules for all clients."""
        print("\n" + "="*60)
        print("RiemannFL Phase 0: Initializing Bidirectional Projections")
        print("="*60)
        
        # Get a sample batch for dimension detection
        sample_batch = next(iter(self.public_loader))
        
        for client in self.clients:
            client_dim = self._get_client_rep_dim(client, sample_batch)
            
            projection = BidirectionalProjection(
                client_dim=client_dim,
                shared_dim=self.shared_dim,
                hidden_dims=self.hidden_dims,
                normalize=True
            ).to(self.device)
            
            self.projections[client.id] = projection
            print(f"  Client {client.id}: d_i={client_dim} → d={self.shared_dim}")
        
        print("="*60 + "\n")
    
    def train(self):
        """
        Train projections using consistency + reconstruction loss.
        
        From Algorithm 1 in the paper.
        """
        from flcore.clients.clientbase import load_item
        
        print("Training bidirectional projections...")
        
        # Collect all projection parameters
        all_params = []
        for proj in self.projections.values():
            all_params.extend(proj.parameters())
        
        optimizer = torch.optim.Adam(all_params, lr=self.init_lr)
        
        # Load models
        models = {}
        for client in self.clients:
            model = load_item(client.role, 'model', client.save_folder_name)
            if model is not None:
                model.eval()
                models[client.id] = model
        
        for epoch in range(self.init_epochs):
            total_loss = 0.0
            total_recon = 0.0
            total_cons = 0.0
            num_batches = 0
            
            for x, _ in self.public_loader:
                x = x.to(self.device)
                optimizer.zero_grad()
                
                # Extract representations for all clients
                reps = {}
                projected = {}
                reconstructed = {}
                
                with torch.no_grad():
                    for cid, model in models.items():
                        try:
                            rep = model.base(x)
                            if rep.dim() == 4:
                                rep = rep.view(rep.size(0), -1)
                            elif rep.dim() == 3:
                                rep = rep.mean(dim=1)
                            reps[cid] = rep.detach()
                        except:
                            continue
                
                # Project and reconstruct (with gradients)
                for cid, rep in reps.items():
                    proj = self.projections[cid]
                    z_tilde, z_rec = proj(rep)
                    projected[cid] = z_tilde
                    reconstructed[cid] = z_rec
                
                if len(projected) < 2:
                    continue
                
                # Reconstruction loss: ||z - Φ⁻¹(Φ(z))||²
                loss_recon = torch.tensor(0.0, device=self.device)
                for cid in projected:
                    loss_recon = loss_recon + F.mse_loss(reconstructed[cid], reps[cid])
                loss_recon = loss_recon / len(projected)
                
                # Consistency loss: KL divergence between projections
                loss_cons = torch.tensor(0.0, device=self.device)
                client_ids = list(projected.keys())
                count = 0
                
                for i, cid_i in enumerate(client_ids):
                    for cid_j in client_ids[i+1:]:
                        p_i = F.softmax(projected[cid_i] / 0.5, dim=-1)
                        p_j = F.softmax(projected[cid_j] / 0.5, dim=-1)
                        
                        kl_ij = F.kl_div(p_i.log(), p_j, reduction='batchmean')
                        kl_ji = F.kl_div(p_j.log(), p_i, reduction='batchmean')
                        loss_cons = loss_cons + (kl_ij + kl_ji) / 2
                        count += 1
                
                if count > 0:
                    loss_cons = loss_cons / count
                
                # Total loss
                loss = self.lambda_recon * loss_recon + self.lambda_cons * loss_cons
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += loss_recon.item()
                total_cons += loss_cons.item()
                num_batches += 1
            
            if num_batches > 0 and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.init_epochs}: "
                      f"Loss={total_loss/num_batches:.4f} "
                      f"(recon={total_recon/num_batches:.4f}, "
                      f"cons={total_cons/num_batches:.4f})")
        
        print("Projection training complete!\n")
    
    def save(self, save_folder: str):
        """Save trained projections."""
        from flcore.clients.clientbase import save_item
        
        for cid, proj in self.projections.items():
            save_item(proj, f'Client_{cid}', 'projection', save_folder)
        
        print(f"Saved {len(self.projections)} bidirectional projections")
    
    def run(self, save_folder: str) -> dict:
        """Run complete initialization phase."""
        self.initialize()
        self.train()
        self.save(save_folder)
        return self.projections