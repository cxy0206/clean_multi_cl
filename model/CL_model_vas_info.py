import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader
from bayes_opt import BayesianOptimization
from tqdm import tqdm
import pandas as pd
import traceback
import os
import itertools
import random
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import math


class FilterEmptyGraphs(torch.utils.data.Dataset):
    """Dataset wrapper to filter out empty graphs"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.filtered_dataset = self._filter_empty_graphs()
    
    def _filter_empty_graphs(self):
        return [data for data in self.dataset 
                if data.x is not None and data.x.shape[0] > 0 
                and data.edge_index is not None and data.edge_index.shape[1] > 0]
    
    def __len__(self):
        return len(self.filtered_dataset)
    
    def __getitem__(self, idx):
        return self.filtered_dataset[idx]


class GNNModelWithNewLoss(nn.Module):
    """GNN model with custom contrastive loss implementation"""
    def __init__(self, num_node_features, num_edge_features, num_global_features, 
                 hidden_dim=256, dropout_rate=0.3, batch_size=512, datasize=False, 
                 device=None, property_index=0, loss_weights={'mse':1, 'rank':0}, save_path="models"):
        super().__init__()
        # Initialize model parameters
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_global_features = num_global_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.datasize = datasize
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.property_index = property_index
        self.loss_weights = loss_weights
        self.save_path = save_path

        # Initialize GAT layers with edge feature handling
        self.conv1 = GATConv(num_node_features, hidden_dim, 
                            edge_dim=num_edge_features, add_self_loops=True)
        self.conv2 = GATConv(hidden_dim, hidden_dim, 
                            edge_dim=num_edge_features, add_self_loops=True)
        self.conv3 = GATConv(hidden_dim, hidden_dim, 
                            edge_dim=num_edge_features, add_self_loops=True)
        
        # Layer normalization
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.bn3 = nn.LayerNorm(hidden_dim)

        self.global_encoder = nn.Linear(num_global_features, 32) if num_global_features > 0 else None
        
        # Projection head for contrastive learning
        if self.num_global_features == 0:
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim//2, 64)
            )
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_dim+32, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim//2, 64)
            )
        
        # Loss calculation strategy
        self.loss_method = "sampling" if datasize else "full_combination"
        self.dropout = nn.Dropout(dropout_rate)

    def get_property(self, batch):
        """Access target property based on stored index"""
        property_name = f"property_{self.property_index}"
        return getattr(batch, property_name, None)

    def forward(self, data):
        """Forward pass with attention weights tracking"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        if self.global_encoder is not None:
            global_embedding = self.global_encoder(data.global_features)
            
        batch = data.batch
        self.attention_weights = []

        # First GAT layer
        x, attn1 = self.conv1(x, edge_index, edge_attr=edge_attr, 
                            return_attention_weights=True)
        self.attention_weights.append(attn1)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)
        
        # Second GAT layer
        x, attn2 = self.conv2(x, edge_index, edge_attr=edge_attr, 
                            return_attention_weights=True)
        self.attention_weights.append(attn2)
        x = F.relu(self.bn2(x))
        x = self.dropout(x)
        
        # Third GAT layer
        x, attn3 = self.conv3(x, edge_index, edge_attr=edge_attr,
                            return_attention_weights=True)
        self.attention_weights.append(attn3)
        x = F.relu(self.bn3(x))
        x = self.dropout(x)
        
        # Global mean pooling
        # graph_embedding = global_mean_pool(x, batch)
        graph_embedding = global_mean_pool(x, batch)

        if self.global_encoder is not None:
            graph_embedding = torch.cat([graph_embedding, global_embedding], dim=1)
        return graph_embedding

    def _project(self, embeddings):
        """Project embeddings through projection head"""
        return self.projection_head(embeddings)

    def get_knn_positive_pairs(self, props, k=10, threshold=0.5):
        """
        get positive pairs based on k-nearest neighbors in the property space
        :param props: [n, d] tensor of properties
        """
        props = props.to(self.device).float() 

        sim_matrix = F.cosine_similarity(props.unsqueeze(1), props.unsqueeze(0), dim=-1)
        dist_matrix = 1 - sim_matrix  

        n = props.size(0)
        dist_matrix.fill_diagonal_(float('inf'))

        topk_dist, topk_idx = torch.topk(dist_matrix, k=k, dim=1, largest=False)

        mask = topk_dist < threshold  
        row_idx = torch.arange(n, device=self.device).unsqueeze(1).expand(-1, k)
        pos_i = row_idx[mask]
        pos_j = topk_idx[mask]
        
        return list(zip(pos_i.tolist(), pos_j.tolist()))
    
    def get_loss(self, batch, temperature=0.1, k=5, vsa_threshold=0.05):
        """
        Compute contrastive loss using VSA-guided positive pairs and distance-based InfoNCE.
        
        Args:
            batch: a mini-batch of graphs
            temperature: scaling factor for contrastive loss
            k: number of nearest neighbors to define positive pairs
            vsa_threshold: maximum property-space distance for positive pairs

        Returns:
            scalar contrastive loss
        """
        if batch.num_graphs < 2:
            return torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            # Extract property vector (e.g., solubility, energy, etc.)
            prop = self.get_property(batch)
            if prop is None:
                return torch.tensor(0.0, device=self.device)
            
            n = prop.shape[0]

            # Use property similarity to define positive pairs
            pos_pairs = self.get_knn_positive_pairs(prop, k=k, threshold=vsa_threshold)
            if len(pos_pairs) == 0:
                return torch.tensor(0.0, device=self.device)

            pos_set = set((i, j) for i, j in pos_pairs)

            # Generate all unique index combinations
            idx_i, idx_j = torch.combinations(torch.arange(n, device=self.device), r=2).unbind(1)
            all_pairs = list(zip(idx_i.tolist(), idx_j.tolist()))

            # Negative pairs = all - positive
            neg_pairs = list(set(all_pairs) - pos_set)

        self.train()
        # Compute projected graph embeddings
        embeddings = self._project(self.forward(batch))  # shape: [n, dim]

        # Compute cosine distances for positive pairs
        pos_i, pos_j = zip(*pos_pairs)
        pos_dist = 1 - F.cosine_similarity(embeddings[list(pos_i)], embeddings[list(pos_j)])
        pos_exp = torch.exp(-pos_dist / temperature)  # Closer → smaller dist → larger exp

        # Compute cosine distances for negative pairs
        if len(neg_pairs) == 0:
            return torch.tensor(0.0, device=self.device)

        neg_i, neg_j = zip(*neg_pairs)
        neg_dist = 1 - F.cosine_similarity(embeddings[list(neg_i)], embeddings[list(neg_j)])
        neg_exp = torch.exp(-neg_dist / temperature)  # Farther → larger dist → smaller exp

        # InfoNCE loss: encourage small dist for positives, large dist for negatives
        numerator = pos_exp.sum()
        denominator = numerator + neg_exp.sum()
        loss = -torch.log(numerator / (denominator + 1e-8))
        with torch.no_grad():
            pos_count = len(pos_pairs)
            neg_count = len(neg_pairs)
            baseline_loss = -math.log(pos_count / (pos_count + neg_count + 1e-8))
            print(f"Baseline Loss: {baseline_loss:.4f} | Actual Loss: {loss.item():.4f}")

        return loss

    def train_model(self, dataset, num_epochs=1000, lr=0.00005, weight_decay=1e-4, 
                    patience=50, batch_size=4096, best_val_loss_all=float('inf')):
        """Training procedure with early stopping"""
        save_path = self.save_path
        print(f"Training will be saved to: {save_path}")
        # Filter empty graphs
        filtered_dataset = FilterEmptyGraphs(dataset)
        train_size = int(0.8 * len(filtered_dataset))
        train_set, val_set = torch.utils.data.random_split(
            filtered_dataset, 
            [train_size, len(filtered_dataset)-train_size]
        )
        # Create data loaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        # Initialize optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Training loop
        for epoch in tqdm(range(1, num_epochs+1), desc="Training"):
            # Training phase
            self.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                loss = self.get_loss(batch)
                
                # Backpropagation with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    val_loss += self.get_loss(batch).item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)
            print(f"Epoch {epoch}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss and avg_val_loss < best_val_loss_all:
                print(f"New best validation loss: {avg_val_loss:.4f}")
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'encoder_state_dict': self.state_dict(),
                }, os.path.join(save_path, "best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        # Final plotting after training completes
        if avg_val_loss < best_val_loss_all:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
            plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Process (Best Val Loss: {best_val_loss:.4f})')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_path, "training_curve.png"))
            plt.close()

        return best_val_loss

    def get_distribution(self, dataloader):
        self.eval()
        save_path = os.path.join(self.save_path, "distribution")
        os.makedirs(save_path, exist_ok=True)
        prop_diffs = []
        combined_dists = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device) 
                
                raw_emb = self.forward(batch).to(self.device)
                proj_emb = self._project(raw_emb).to(self.device)
                
                prop = self.get_property(batch).to(self.device)  
                
                n = batch.num_graphs

                if n > 1:
                    i, j = torch.combinations(torch.arange(n, device=self.device), 2).unbind(1)

                    if prop.shape[1] == 1:
                        _prop_diff = torch.abs(prop[i] - prop[j])
                    else:
                        _prop_diff = 1 - F.cosine_similarity(prop[i], prop[j])

                    _cos_dist = 1 - F.cosine_similarity(proj_emb[i], proj_emb[j])

                    if torch.any(torch.isnan(_prop_diff)) or torch.any(torch.isnan(_cos_dist)):
                        print("NaN detected, skipping this batch.")
                        continue

                    prop_diffs.append(_prop_diff.cpu().numpy())
                    combined_dists.append(_cos_dist.cpu().numpy())
                else:
                    print(f"Skipping batch with only {n} graph(s)")

        if prop_diffs and combined_dists:
            prop_diffs = np.concatenate(prop_diffs)
            combined_dists = np.concatenate(combined_dists)

            plt.figure(figsize=(10,6))
            plt.scatter(prop_diffs, combined_dists, alpha=0.6, edgecolors='w', linewidth=0.5)
            plt.plot([0,1], [0,1], 'r--', linewidth=2)
            plt.xlabel('Property Difference')
            plt.ylabel('Embedding Distance')
            plt.title(f'Validation Set: PropDiff vs EmbedDist')
            plt.grid(True)
            plt.savefig(os.path.join(save_path, "scatter_plot.png"))
            plt.close()
        else:
            print("No valid data to plot.")

