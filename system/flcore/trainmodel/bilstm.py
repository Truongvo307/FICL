# https://github.com/FedML-AI/FedNLP/blob/master/model/bilstm.py
import torch
from torch import nn
import torch.nn.functional as F

class BiLSTM_TextClassification(nn.Module):
    def __init__(
        self,
        input_size,          # vocab_size
        hidden_size,         # feature_dim / embedding dim
        output_size,         # num_classes
        num_layers=1,
        embedding_dropout=0.0,
        lstm_dropout=0.0,
        attention_dropout=0.0,
        fc_dropout=0.0,      # ← add this param
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.emb_drop  = nn.Dropout(embedding_dropout)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        self.attn_drop = nn.Dropout(attention_dropout)
        self.fc_drop   = nn.Dropout(fc_dropout)
        # BiLSTM → hidden_size * 2; attention-pooled → same
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def _attention(self, lstm_out):
        """Additive self-attention over the time dimension."""
        # lstm_out: [B, T, H*2]
        scores = torch.tanh(lstm_out)          # [B, T, H*2]
        scores = scores.sum(dim=-1, keepdim=True)  # [B, T, 1]
        weights = torch.softmax(scores, dim=1)     # [B, T, 1]
        weights = self.attn_drop(weights)
        context = (weights * lstm_out).sum(dim=1)  # [B, H*2]
        return context

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            text, text_lengths = x
        else:
            text = x
            text_lengths = [x.shape[1]] * x.shape[0]

        emb = self.emb_drop(self.embedding(text))          # [B, T, H]

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, text_lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: [B, T, H*2]

        context = self._attention(out)                     # [B, H*2]
        context = self.fc_drop(context)
        logits  = self.fc(context)                         # [B, num_classes]
        return F.log_softmax(logits, dim=1)