import torch
import torch.nn as nn
import math

class TransformerFusionModel(nn.Module):
    def __init__(self, emb_dim, hidden_dim=512,global_dim=32):
        super().__init__()
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.query  = nn.Parameter(torch.randn(emb_dim))
        self.global_encoder = nn.Linear(8, global_dim)  # Assuming global features are of size 8
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim+global_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x,global_features):
        B, N, D = x.size()
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = self.query.unsqueeze(0).unsqueeze(1).expand(B, 1, D)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        weights = torch.softmax(scores, dim=-1)
        fused = torch.matmul(weights, V).squeeze(1)
        global_features = global_features.squeeze(1)
        fused = torch.cat([fused, self.global_encoder(global_features)], dim=-1)  # [B, D + G]
        out = self.mlp(fused).squeeze(-1)
        return out, weights.squeeze(1)

class WeightedFusion(nn.Module):
    def __init__(self, num_inputs=3, emb_dim=512, dropout=0.1, layer_norm_out=True, global_dim=32):
    
        super().__init__()
        self.emb_dim = emb_dim
        self.num_inputs = num_inputs
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.weight_logits = nn.Parameter(torch.zeros(num_inputs))  # initialized to uniform weights
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim) if layer_norm_out else None
        self.global_encoder = nn.Linear(8, global_dim)  # Assuming global features are of size 8
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim+global_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, embs, global_features):  # embs: [B, N, D]
        B, N, D = embs.size()
        x = self.linear(embs)  # shape [B, N, D]
        norm_weights = torch.softmax(self.weight_logits, dim=0)  # shape [N]
        fused = torch.einsum('bnd,n->bd', x, norm_weights)  # [B, D]
        fused = self.dropout(fused)
        if self.layer_norm is not None:
            fused = self.layer_norm(fused)

        global_features = global_features.squeeze(1)
        fused = torch.cat([fused, self.global_encoder(global_features)], dim=-1)  # [B, D + G]

        out = self.mlp(fused).squeeze(-1)
        return out, norm_weights
        
class MLP(nn.Module):
    def __init__(self, emb_dim, hidden_dim=64,global_dim=32):
        super().__init__()
        self.global_encoder = nn.Linear(8, global_dim)  # Assuming global features are of size 8
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim+global_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x, global_features):
        global_features = global_features.squeeze(1)
        x = torch.cat([x, self.global_encoder(global_features)], dim=-1)  # [B, D + G]
        out = self.mlp(x).squeeze(-1)
        return out


# fine-tune model
class FusionFineTuneModel(nn.Module):
    def __init__(self, encoder_list, fusion_model, fusion_method='attention'):
        super().__init__()
        self.encoders = nn.ModuleList(encoder_list)
        self.fusion = fusion_model
        self.fusion_method = fusion_method

    def forward(self, data):
        embs = [encoder(data) for encoder in self.encoders]  # list of [B, D]
        embs = torch.stack(embs, dim=1)  # [B, 3, D]
        if self.fusion_method == 'attention':
            out, weights = self.fusion(embs)
            return out, weights
        else:
            weights = 0
            out = self.fusion(torch.cat([embs[:, i, :] for i in range(embs.size(1))], dim=1))
            return out, weights
