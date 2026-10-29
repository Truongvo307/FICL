"""
Optimized Projection Module for RiemannFL

Key optimizations:
1. Simpler architecture (less parameters)
2. No reconstruction loss during main training
3. Faster initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
from flcore.trainmodel.etf_anchors import build_etf_anchors


@staticmethod
def compute_anchor_stats(z_tilde: torch.Tensor):
        # z_tilde: (B, d)
        mu = z_tilde.mean(dim=0)
        var = z_tilde.var(dim=0, unbiased=False) + 1e-6
        return mu, var
@staticmethod
def kl_diag_gaussians(mu_i, var_i, mu_a, var_a):
        # KL(q_i || q_a) for diagonal Gaussians
        return 0.5 * torch.sum(
            var_a / var_i + var_i / var_a + (mu_i - mu_a) ** 2 / var_a - 1.0
        )

@staticmethod
def info_max_loss(z_tilde: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Information maximization: -log det(covariance).
    z_tilde: (B, d)
    """
    z_centered = z_tilde - z_tilde.mean(dim=0)
    cov = (z_centered.t() @ z_centered) / max(1, z_tilde.shape[0] - 1)
    cov = cov + eps * torch.eye(cov.shape[0], device=z_tilde.device)
    eigvals = torch.linalg.eigvalsh(cov)
    return -torch.sum(torch.log(eigvals + eps))

@staticmethod
def _vicreg_anchor_loss(
    z_tilde: torch.Tensor,
    y: torch.Tensor,
    anchors: torch.Tensor,
    gamma: float,
    lambda_cov: float,
    use_anchor_loss: bool
) -> tuple:
    """
    Shared helper: computes (L_anc, L_reg) given a batch of projected
    features z_tilde [B, d] and labels y [B].
    """
    device = z_tilde.device
    classes_in_batch = y.unique()

    # --- L_anc: cosine pull toward ETF anchors ---
    loss_anc = torch.tensor(0.0, device=device)
    if use_anchor_loss and len(classes_in_batch) > 0:
        for c in classes_in_batch:
            mask = (y == c)
            zbar_c = F.normalize(z_tilde[mask].mean(dim=0, keepdim=True), p=2, dim=1)
            loss_anc = loss_anc + (1.0 - (zbar_c * anchors[c.item()]).sum())
        loss_anc = loss_anc / len(classes_in_batch)

    # --- L_reg: VICReg-style variance + covariance ---
    var_term = torch.tensor(0.0, device=device)
    valid_classes = 0
    for c in classes_in_batch:
        mask = (y == c)
        if mask.sum() > 1:
            std_c = z_tilde[mask].std(dim=0).mean()
            var_term = var_term + F.relu(gamma - std_c)
            valid_classes += 1
    if valid_classes > 0:
        var_term = var_term / valid_classes

    z_centered = z_tilde - z_tilde.mean(dim=0)
    B = z_tilde.shape[0]
    cov = (z_centered.T @ z_centered) / max(B - 1, 1)
    cov_off = cov - torch.diag(torch.diag(cov))
    cov_term = lambda_cov * (cov_off ** 2).sum()

    loss_reg = var_term + cov_term

    return loss_anc, loss_reg


# ---------------------------------------------------------------------------
# Phase-0 loss helpers
# ---------------------------------------------------------------------------
@staticmethod
def phase0_loss(
    z_tilde: torch.Tensor,
    z_rec: torch.Tensor,
    z_orig: torch.Tensor,
    labels: torch.Tensor,
    anchors: torch.Tensor,
    lambda_rec: float = 1.0,
    lambda_anc: float = 0.1,
    use_varcov_reg: bool = False,   # ABLATION-ONLY, default False
    lambda_reg: float = 1.0,        # ablation only
    lambda_cov: float = 1.0,        # ablation only
    gamma: float = 1.0,             # ablation only
) -> torch.Tensor:
    """
    GeoProtoFL Phase-0 objective (two terms in default path):
        L0 = lambda_rec * L_rec  +  lambda_anc * L_anc

    Args:
        z_tilde : [B, d] L2-normalized encoder output (on S^{d-1})
        z_rec   : [B, client_dim] decoder reconstruction
        z_orig  : [B, client_dim] original representation (target for rec)
        labels  : [B] integer class labels
        anchors : [C, d] ETF anchors (from build_etf_anchors)
        lambda_rec : fixed reference scale = 1.0
        lambda_anc : the ONE tunable weight (default 0.1)
        use_varcov_reg : ABLATION-ONLY switch (False by default)
    """
    # --- L_rec: reconstruction (identifiability + anti-collapse) ---
    L_rec = F.mse_loss(z_rec, z_orig)

    # --- L_anc: ETF anchor loss (gauge fixing + spread) ---
    unique_classes = labels.unique()
    L_anc = torch.tensor(0.0, device=z_tilde.device)
    n_cls = 0

    for c in unique_classes:
        c_idx = c.item()
        if c_idx >= anchors.shape[0]:
            continue
        mask = (labels == c)
        z_c = z_tilde[mask]                                       # [n_c, d]
        zbar_c = F.normalize(z_c.mean(dim=0), dim=0)             # [d]
        cosine = (zbar_c * anchors[c_idx].to(z_tilde.device)).sum()
        L_anc = L_anc + (1.0 - cosine)
        n_cls += 1

    if n_cls > 0:
        L_anc = L_anc / n_cls



    L0 = lambda_rec * L_rec + lambda_anc * L_anc
    # [ABLATION-ONLY] var/cov regularizer — NOT in default path
    if use_varcov_reg:
        var_term  = torch.tensor(0.0, device=z_tilde.device)
        cov_matrix = torch.cov(z_tilde.T)
        diag_cov   = torch.diag(torch.diag(cov_matrix))
        cov_term   = torch.norm(cov_matrix - diag_cov, p='fro') ** 2

        for c in unique_classes:
            mask = (labels == c)
            if mask.sum() < 2:
                continue
            std_c = z_tilde[mask].std(dim=0).mean()
            var_term = var_term + F.relu(gamma - std_c)

        L0 = L0 + lambda_reg * (var_term + lambda_cov * cov_term)

    return L0


class BidirectionalProjection(nn.Module):
    """
    Lightweight bidirectional projection.
    
    Simplified architecture for faster training.
    """
    
    def __init__(
        self,
        client_dim: int,
        shared_dim: int,
        hidden_dim: int = 256,  # Single hidden layer
        normalize: bool = True
    ):
        super().__init__()
        
        self.client_dim = client_dim
        self.shared_dim = shared_dim
        self.normalize = normalize
        
        # Encoder: client_dim → hidden → shared_dim
        self.encoder = nn.Sequential(
            nn.Linear(client_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, shared_dim)
        )
        
        # Decoder: shared_dim → hidden → client_dim
        self.decoder = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, client_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_project(self, z: torch.Tensor) -> torch.Tensor:
        """Encode to shared space."""
        z_tilde = self.encoder(z)
        if self.normalize:
            z_tilde = F.normalize(z_tilde, p=2, dim=-1)
        return z_tilde
    
    def inverse_project(self, w: torch.Tensor) -> torch.Tensor:
        """Decode back to client space."""
        return self.decoder(w)
    
    def forward(self, z: torch.Tensor) -> tuple:
        """Full pass for reconstruction loss."""
        z_tilde = self.forward_project(z)
        z_rec = self.inverse_project(z_tilde)
        return z_tilde, z_rec
    




class LocalProjectionTrainer:
    """    
    Phase-0 local projection training.
    Objective: L0 = lambda_rec * L_rec + lambda_anc * L_anc
    No prototype or weight exchange during Phase 0.
    """
    
    def __init__(
        self,
        client,
        shared_dim: int,
        device: torch.device,
        p0_epochs: int = 5,  # Reduced
        p0_lr: float = 0.01,
        lambda_recon: float = 1.0,
        hidden_dim: int = 256,
        lambda_info: float = 0.1,
        lambda_rec: float = 1.0, 
        lambda_anc: float = 0.1,
        use_varcov_reg: bool = False,
    ):
        self.client = client
        self.shared_dim = shared_dim
        self.device = device
        self.p0_epochs = p0_epochs
        self.p0_lr = p0_lr
        self.lambda_recon = lambda_recon
        self.hidden_dim = hidden_dim
        self.lambda_info = lambda_info
        self.lambda_rec = lambda_rec
        self.lambda_anc = lambda_anc
        self.use_varcov_reg = use_varcov_reg
        self.num_classes = client.num_classes

    def train(self) -> BidirectionalProjection:
        """Train projection with early stopping."""
        from flcore.clients.clientbase import load_item, save_item
        from flcore.trainmodel.etf_anchors import build_etf_anchors

        model = load_item(self.client.role, 'model', self.client.save_folder_name)
        if model is None:
            raise ValueError(f"Model not found for {self.client.role}")
        
        trainloader = self.client.load_train_data()
        
        # Get representation dimension
        client_dim = self._get_rep_dim(model, trainloader)
        num_classes = self.client.num_classes

        anchors = build_etf_anchors(num_classes, self.shared_dim).to(self.device)


        # Create lightweight projection
        projection = BidirectionalProjection(
            client_dim=client_dim,
            shared_dim=self.shared_dim,
            hidden_dim=self.hidden_dim,
            normalize=True
        ).to(self.device)
        
        optimizer = torch.optim.Adam(projection.parameters(), lr=self.p0_lr)
        
        model.eval()
        projection.train()
        
        print(f"  Client {self.client.id}: d_i={client_dim} → d={self.shared_dim}")
        
        prev_loss = float('inf')
        patience = 2
        patience_counter = 0
        
        for epoch in range(self.p0_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for x, y in trainloader:
                if isinstance(x, (list, tuple)):
                    # x = [xi.to(self.device) if isinstance(xi, torch.Tensor) else xi for xi in x]
                    x = [x[0].to(self.device)] + list(x[1:]) 
                else:
                    x = x.to(self.device)
                
                y = y.to(self.device)
                optimizer.zero_grad()
                
                with torch.no_grad():
                    rep = model.base(x)
                    if rep.dim() == 4:
                        rep = rep.view(rep.size(0), -1)
                    elif rep.dim() == 3:
                        rep = rep.mean(dim=1)
                
                z_tilde, z_rec = projection(rep)

                loss = phase0_loss(
                    z_tilde        = z_tilde,
                    z_rec          = z_rec,
                    z_orig         = rep,
                    labels         = y,
                    anchors        = anchors,
                    lambda_rec     = self.lambda_rec,
                    lambda_anc     = self.lambda_anc,
                    use_varcov_reg = self.use_varcov_reg,
                )
                
                
                # loss = self.lambda_recon * F.mse_loss(z_rec, rep)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            
            # Early stopping
            if avg_loss >= prev_loss * 0.99:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stop at epoch {epoch+1}")
                    break
            else:
                patience_counter = 0
            prev_loss = avg_loss
        
        save_item(projection, self.client.role, 'projection', self.client.save_folder_name)
        return projection
    
    def _get_rep_dim(self, model, loader) -> int:
        model.eval()
        with torch.no_grad():
            for x, _ in loader:
                if isinstance(x, (list, tuple)):
                    # x = [xi.to(self.device) if isinstance(xi, torch.Tensor) else xi for xi in x]
                    x = [x[0].to(self.device)] + list(x[1:])  # text -> GPU, lengths stay on CPU
                else:
                    x = x.to(self.device)
                rep = model.base(x[:2])
                if rep.dim() == 4:
                    rep = rep.view(rep.size(0), -1)
                elif rep.dim() == 3:
                    rep = rep.mean(dim=1)
                return rep.shape[-1]
        return 512


class GlobalProjectionTrainer:
    """
    Global projection training with consistency loss.
    
    Optimized for speed while maintaining alignment.
    """
    
    def __init__(
        self,
        clients: list,
        shared_dim: int,
        device: torch.device,
        public_data_loader,
        p0_epochs: int = 10,
        p0_lr: float = 0.01,
        lambda_recon: float = 1.0,
        lambda_cons: float = 0.5,
        hidden_dim: int = 256
    ):
        self.clients = clients
        self.shared_dim = shared_dim
        self.device = device
        self.public_loader = public_data_loader
        self.p0_epochs = p0_epochs
        self.p0_lr = p0_lr
        self.lambda_recon = lambda_recon
        self.lambda_cons = lambda_cons
        self.hidden_dim = hidden_dim
        self.projections = {}
    
    def train(self) -> dict:
        """Train with consistency + reconstruction loss."""
        from flcore.clients.clientbase import load_item, save_item
        
        print("  Global mode: Training with consistency loss")
        
        # Initialize projections
        sample_batch = next(iter(self.public_loader))
        
        for client in self.clients:
            client_dim = self._get_rep_dim(client, sample_batch)
            
            projection = BidirectionalProjection(
                client_dim=client_dim,
                shared_dim=self.shared_dim,
                hidden_dim=self.hidden_dim,
                normalize=True
            ).to(self.device)
            
            self.projections[client.id] = projection
            print(f"    Client {client.id}: d_i={client_dim}")
        
        # Collect parameters
        all_params = []
        for proj in self.projections.values():
            all_params.extend(proj.parameters())
        
        optimizer = torch.optim.Adam(all_params, lr=self.p0_lr)
        
        # Load models
        models = {}
        for client in self.clients:
            model = load_item(client.role, 'model', client.save_folder_name)
            if model is not None:
                model.eval()
                models[client.id] = model
        
        for epoch in range(self.p0_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for x, _ in self.public_loader:
                x = x.to(self.device)
                optimizer.zero_grad()
                
                # Extract and project
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
                
                for cid, rep in reps.items():
                    z_tilde, z_rec = self.projections[cid](rep)
                    projected[cid] = z_tilde
                    reconstructed[cid] = z_rec
                
                if len(projected) < 2:
                    continue
                
                # Reconstruction loss
                loss_recon = sum(
                    F.mse_loss(reconstructed[cid], reps[cid])
                    for cid in projected
                ) / len(projected)
                
                # Consistency loss (align projections)
                loss_cons = torch.tensor(0.0, device=self.device)
                client_ids = list(projected.keys())
                
                # Compare to mean (faster than pairwise)
                mean_proj = torch.stack(list(projected.values())).mean(dim=0)
                for cid in client_ids:
                    loss_cons = loss_cons + F.mse_loss(projected[cid], mean_proj.detach())
                loss_cons = loss_cons / len(client_ids)
                
                loss = self.lambda_recon * loss_recon + self.lambda_cons * loss_cons
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}: loss={total_loss/max(num_batches,1):.4f}")
        
        # Save
        for client in self.clients:
            save_item(
                self.projections[client.id],
                client.role,
                'projection',
                client.save_folder_name
            )
        
        return self.projections
    
    def _get_rep_dim(self, client, sample_batch) -> int:
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


class FederatedProjectionTrainer:
    """
    Federated projection training.
    
    Simplified: fewer rounds, faster convergence.
    """
    
    def __init__(
        self,
        clients: list,
        shared_dim: int,
        device: torch.device,
        p0_rounds: int = 3,  # Reduced
        p0_local_epochs: int = 2,
        p0_lr: float = 0.01,
        lambda_recon: float = 1.0,
        hidden_dim: int = 256
    ):
        self.clients = clients
        self.shared_dim = shared_dim
        self.device = device
        self.p0_rounds = p0_rounds
        self.p0_local_epochs = p0_local_epochs
        self.p0_lr = p0_lr
        self.lambda_recon = lambda_recon
        self.hidden_dim = hidden_dim
        self.projections = {}
    
    def train(self) -> dict:
        """Federated training with encoder aggregation."""
        from flcore.clients.clientbase import load_item, save_item
        
        print(f"  Federated: {self.p0_rounds} rounds")
        
        # Initialize
        self._init_projections()
        
        for r in range(self.p0_rounds):
            # Local training
            for client in self.clients:
                self._train_local(client)
            
            # Aggregate encoders
            self._aggregate()
            
            print(f"    Round {r+1}/{self.p0_rounds} done")
        
        # Save
        for client in self.clients:
            save_item(
                self.projections[client.id],
                client.role,
                'projection',
                client.save_folder_name
            )
        
        return self.projections
    
    def _init_projections(self):
        from flcore.clients.clientbase import load_item
        
        for client in self.clients:
            model = load_item(client.role, 'model', client.save_folder_name)
            loader = client.load_train_data()
            
            client_dim = self._get_rep_dim(model, loader)
            
            projection = BidirectionalProjection(
                client_dim=client_dim,
                shared_dim=self.shared_dim,
                hidden_dim=self.hidden_dim,
                normalize=True
            ).to(self.device)
            
            self.projections[client.id] = projection
            print(f"    Client {client.id}: d_i={client_dim}")
    
    def _get_rep_dim(self, model, loader) -> int:
        model.eval()
        with torch.no_grad():
            for x, _ in loader:
                if isinstance(x, (list, tuple)):
                    # x = [xi.to(self.device) if isinstance(xi, torch.Tensor) else xi for xi in x]
                    x = [x[0].to(self.device)] + list(x[1:]) 
                else:
                    x = x.to(self.device)
                rep = model.base(x[:2])
                if rep.dim() == 4:
                    rep = rep.view(rep.size(0), -1)
                elif rep.dim() == 3:
                    rep = rep.mean(dim=1)
                return rep.shape[-1]
        return 512
    
    def _train_local(self, client):
        from flcore.clients.clientbase import load_item
        
        model = load_item(client.role, 'model', client.save_folder_name)
        projection = self.projections[client.id]
        loader = client.load_train_data()
        
        optimizer = torch.optim.Adam(projection.parameters(), lr=self.p0_lr)
        
        model.eval()
        projection.train()
        
        for _ in range(self.p0_local_epochs):
            for x, _ in loader:
                if isinstance(x, (list, tuple)):
                    # x = [xi.to(self.device) if isinstance(xi, torch.Tensor) else xi for xi in x]
                    x = [x[0].to(self.device)] + list(x[1:]) 
                else:
                    x = x.to(self.device)
                
                optimizer.zero_grad()
                
                with torch.no_grad():
                    rep = model.base(x)
                    if rep.dim() == 4:
                        rep = rep.view(rep.size(0), -1)
                    elif rep.dim() == 3:
                        rep = rep.mean(dim=1)
                
                _, z_rec = projection(rep)
                loss = self.lambda_recon * F.mse_loss(z_rec, rep)
                
                loss.backward()
                optimizer.step()
    
    def _aggregate(self):
        """Aggregate encoder's last layer (same dimensions)."""
        # Get last layer params (shared_dim output)
        last_layer_params = {}
        
        for cid, proj in self.projections.items():
            # encoder[-1] is the last Linear layer
            last_layer = proj.encoder[-1]
            last_layer_params[cid] = {
                'weight': last_layer.weight.data.clone(),
                'bias': last_layer.bias.data.clone()
            }
        
        # Average
        avg_weight = torch.stack([p['weight'] for p in last_layer_params.values()]).mean(0)
        avg_bias = torch.stack([p['bias'] for p in last_layer_params.values()]).mean(0)
        
        # Update
        for proj in self.projections.values():
            proj.encoder[-1].weight.data = avg_weight.clone()
            proj.encoder[-1].bias.data = avg_bias.clone()



    