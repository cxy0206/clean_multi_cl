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

class GNNModelWithContrastiveLearning(nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_global_features, hidden_dim=64, dropout_rate=0.3,batch_size=32,device='cpu',is_anchor_based=True):
        super(GNNModelWithContrastiveLearning, self).__init__()

        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_global_features = num_global_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.device = device
        self.is_anchor_based = is_anchor_based

        # GNN layers
        self.conv1 = GATConv(num_node_features, hidden_dim, add_self_loops=True)
        self.conv2 = GATConv(hidden_dim, hidden_dim, add_self_loops=True)
        self.conv3 = GATConv(hidden_dim, hidden_dim, add_self_loops=True)
        
        # Batch normalization layers
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.bn3 = nn.LayerNorm(hidden_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        if data.x.shape[0] == 0 or data.edge_index.shape[1] == 0:
            print("Warning: Empty graph in batch!")
        device = data.x.device
        x, edge_index, edge_attr, global_features = data.x, data.edge_index, data.edge_attr, data.global_features

        # Apply GAT layers with BatchNorm and Dropout
        x, self.att1 = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x, self.att2 = self.conv2(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x, self.att3 = self.conv3(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Global pooling: aggregate node features to a single graph-level feature
        graph_embedding = global_mean_pool(x, data.batch)

        return graph_embedding

    def cosine_similarity(self, x, y):
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        return torch.matmul(x_norm, y_norm.T)
    
    def info_nce_loss(self, anchor, positive_list, negative_list, temperature):
        # Ensure that both positive_list and negative_list are non-empty
        if not positive_list :
            raise ValueError("Positive list is empty, cannot compute loss.")
        
        if not negative_list :
            raise ValueError("Negative list is empty, cannot compute loss.")

        # Calculate the cosine similarity between anchor and positive samples
        anchor_positive_sims = []
        
        for positive in positive_list:
            sim = torch.cosine_similarity(anchor, positive, dim=0)
            anchor_positive_sims.append(sim)
        
        # Calculate the cosine similarity between anchor and negative samples
        anchor_negative_sims = []
        for negative in negative_list:
            sim = torch.cosine_similarity(anchor, negative, dim=0)
            anchor_negative_sims.append(sim)

        # Ensure there are no empty lists before performing the stack operation
        if len(anchor_positive_sims) == 0 or len(anchor_negative_sims) == 0:
            raise ValueError("Anchor-positive or anchor-negative similarity lists are empty.")
        
        # Calculate exponentiated similarities
        anchor_positive_exp_sims = [torch.exp(sim / temperature) for sim in anchor_positive_sims]
        anchor_negative_exp_sims = [torch.exp(sim / temperature) for sim in anchor_negative_sims]
        
        positive_pair_exp_sims = []
        
        for i in range(len(positive_list)):
            for j in range(i + 1, len(positive_list)):
                sim = torch.cosine_similarity(positive_list[i], positive_list[j], dim=0) / temperature
                sim = torch.exp(sim)
                positive_pair_exp_sims.append(sim)

        positive_negative_exp_sims = []
        for positive in positive_list:
            for negative in negative_list:
                sim = torch.cosine_similarity(positive, negative, dim=0) / temperature
                sim = torch.exp(sim)
                positive_negative_exp_sims.append(sim)
        
        # Stack the similarities if the lists are non-empty
        if positive_pair_exp_sims:
            positive_exp_sim_sum = torch.sum(torch.stack(positive_pair_exp_sims), dim=0)
        else:
            positive_exp_sim_sum = torch.tensor(0.0).to(anchor.device)
        
        if positive_negative_exp_sims:
            positive_negative_exp_sim_sum = torch.sum(torch.stack(positive_negative_exp_sims), dim=0)
        else:
            positive_negative_exp_sim_sum = torch.tensor(0.0).to(anchor.device)

        if anchor_positive_exp_sims:
            anchor_positive_exp_sim_sum = torch.sum(torch.stack(anchor_positive_exp_sims), dim=0)
        else:
            anchor_positive_exp_sim_sum = torch.tensor(0.0).to(anchor.device)
        
        if anchor_negative_exp_sims:
            anchor_negative_exp_sim_sum = torch.sum(torch.stack(anchor_negative_exp_sims), dim=0)
        else:
            anchor_negative_exp_sim_sum = torch.tensor(0.0).to(anchor.device)
        
        # Final loss calculation
        loss = -torch.log((anchor_positive_exp_sim_sum+positive_exp_sim_sum) / (positive_exp_sim_sum+anchor_positive_exp_sim_sum + positive_negative_exp_sim_sum+anchor_negative_exp_sim_sum+ 1e-8))
        return loss.mean()
    
    def info_nce_loss_withoutAnchor(self, positive_pairs, negative_pairs, temperature):
        """
        Contrastive loss function for anchor-free approach.
        
        Args:
            positive_pairs: List of tuples (data1, data2) representing positive pairs. Each data is a DataBatch.
            negative_pairs: List of tuples (data1, data2) representing negative pairs. Each data is a DataBatch.
            temperature: Float, temperature parameter for scaling.
        
        Returns:
            loss: Scalar tensor representing the loss.
        """
        if len(positive_pairs) == 0 or len(negative_pairs) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Compute embeddings for all positive pairs
        positive_embeddings1 = []
        positive_embeddings2 = []
        for pair in positive_pairs:
            data1, data2 = pair
            embedding1 = self(data1)  # Forward pass through the model
            embedding2 = self(data2)
            
            if embedding1.dim() == 2 and embedding1.size(0) == 1:
                embedding1 = embedding1.squeeze(0)
            if embedding2.dim() == 2 and embedding2.size(0) == 1:
                embedding2 = embedding2.squeeze(0)
            
            positive_embeddings1.append(embedding1)  # Shape: [embedding_dim]
            positive_embeddings2.append(embedding2)
        
        # Compute embeddings for all negative pairs
        negative_embeddings1 = []
        negative_embeddings2 = []
        for pair in negative_pairs:
            data1, data2 = pair
            embedding1 = self(data1)
            embedding2 = self(data2)
            
            if embedding1.dim() == 2 and embedding1.size(0) == 1:
                embedding1 = embedding1.squeeze(0)
            if embedding2.dim() == 2 and embedding2.size(0) == 1:
                embedding2 = embedding2.squeeze(0)
            
            negative_embeddings1.append(embedding1)
            negative_embeddings2.append(embedding2)
        
        # Stack embeddings into tensors
        try:
            positive_embeddings1 = torch.stack(positive_embeddings1)  # Shape: [num_positive_pairs, embedding_dim]
            positive_embeddings2 = torch.stack(positive_embeddings2)  # Shape: [num_positive_pairs, embedding_dim]
            negative_embeddings1 = torch.stack(negative_embeddings1)  # Shape: [num_negative_pairs, embedding_dim]
            negative_embeddings2 = torch.stack(negative_embeddings2)  # Shape: [num_negative_pairs, embedding_dim]
        except RuntimeError as e:
            print(f"Error stacking embeddings: {e}")
            return torch.tensor(0.0, device=self.device)
        
        # Compute cosine similarities
        # Positive pairs similarity
        pos_sims = F.cosine_similarity(positive_embeddings1, positive_embeddings2, dim=-1) / temperature  # Shape: [num_positive_pairs]
        pos_exp = torch.exp(pos_sims)  # Shape: [num_positive_pairs]
        
        # Negative pairs similarity
        neg_sims = F.cosine_similarity(negative_embeddings1, negative_embeddings2, dim=-1) / temperature  # Shape: [num_negative_pairs]
        neg_exp = torch.exp(neg_sims)  # Shape: [num_negative_pairs]
        
        # Sum of positive similarities
        numerator = torch.sum(pos_exp)  # Scalar
        
        # Sum of all similarities
        denominator = numerator + torch.sum(neg_exp)  # Scalar
        
        # Compute loss
        loss = -torch.log(numerator / (denominator + 1e-8))  # Scalar
        
        return loss


    def validate_model(self, val_loader, temperature=0.1, device='cpu',):
        """
        Validate the model on the validation set, handling both anchor-based and anchor-free cases.
        
        Args:
            val_loader: DataLoader for the validation set.
            temperature: Float, temperature parameter for scaling.
            device: Device to run the validation on ('cpu' or 'cuda').
            use_anchor: Boolean, whether the dataset is anchor-based.
        
        Returns:
            avg_loss: Average validation loss.
        """
        self.eval()  # Set the model to evaluation mode
        total_loss = 0
        use_anchor = self.is_anchor_based

        with torch.no_grad():  # Disable gradient computation
            for data in val_loader:
                if use_anchor:
                    # Anchor-based: (anchor_data, positive_list, negative_list)
                    anchor_data, positive_list, negative_list = data
                    anchor_data = anchor_data.to(device)
                    positive_data = [positive.to(device) for positive in positive_list]
                    negative_data = [negative.to(device) for negative in negative_list]

                    try:
                        # Get the embeddings for the anchor and positive/negative samples
                        anchor_embedding = self(anchor_data)  # Shape: (batch_size, embedding_dim)
                        positive_embeddings = [self(positive) for positive in positive_data]  # List of tensors
                        negative_embeddings = [self(negative) for negative in negative_data]  # List of tensors

                        # Iterate over each sample in the batch
                        for anchor, positives, negatives in zip(anchor_embedding, positive_embeddings, negative_embeddings):
                            # Compute loss for each anchor with its positives and negatives
                            loss = self.info_nce_loss(anchor, positives, negatives, temperature)
                            total_loss += loss.item()
                    except Exception as e:
                        # If an error occurs, print the error and skip this batch
                        print(f"Skipping batch due to error: {e}")
                        continue
                else:
                    # Anchor-free: (positive_pairs, negative_pairs)
                    positive_pairs, negative_pairs = data
                    # Move all pairs to the specified device
                    positive_pairs = [(pair[0].to(device), pair[1].to(device)) for pair in positive_pairs]
                    negative_pairs = [(pair[0].to(device), pair[1].to(device)) for pair in negative_pairs]

                    try:
                        # Compute loss for anchor-free case
                        loss = self.info_nce_loss_withoutAnchor(positive_pairs, negative_pairs, temperature)
                        total_loss += loss.item()
                    except Exception as e:
                        print(f"Skipping batch due to error: {e}")
                        continue

        # Calculate the average loss
        avg_loss = total_loss / len(val_loader)
        return avg_loss

    
    def train_model(self, dataset, num_epochs=50, lr=0.001, weight_decay=1e-4, temperature=0.1, device='cpu', patience=20, batch_size=32):
        """
        Trains the GNN model with contrastive learning, handling both anchor-based and anchor-free cases,
        including progress bars for epochs and batches, and optional early stopping.

        Parameters:
        - dataset: The dataset to train on.
        - num_epochs: Number of training epochs.
        - lr: Learning rate.
        - weight_decay: Weight decay (L2 regularization).
        - temperature: Temperature parameter for InfoNCE loss.
        - device: Device to train on ('cpu' or 'cuda').
        - patience: Number of epochs to wait for improvement in validation loss before stopping early.
        - batch_size: Batch size for training and validation.
        - use_anchor: Boolean flag to indicate whether to use anchor-based or anchor-free contrastive learning.

        Returns:
        - avg_train_loss: Average training loss of the last epoch.
        - train_losses: List of average training losses per epoch.
        - val_losses: List of average validation losses per epoch.
        """
        # Split the dataset into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        use_anchor = self.is_anchor_based

        # Initialize the model
        model = GNNModelWithContrastiveLearning(
            num_node_features=self.num_node_features, 
            num_edge_features=self.num_edge_features, 
            num_global_features=self.num_global_features, 
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,  
            device=device,
            is_anchor_based=use_anchor
        ).to(device)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Lists to store the losses
        train_losses = []
        val_losses = []

        # Variables for early stopping
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        # Training loop with tqdm progress bars
        for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs", unit="epoch"):
            # Training phase
            model.train()
            total_train_loss = 0

            # Initialize a tqdm progress bar for batches
            batch_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False, unit="batch")
            for batch in batch_bar:
                if use_anchor:
                    # Anchor-based: (anchor_data, positive_list, negative_list)
                    anchor_data, positive_list, negative_list = batch
                    # Move data to the specified device
                    anchor_data = anchor_data.to(device)
                    positive_data = [positive.to(device) for positive in positive_list]
                    negative_data = [negative.to(device) for negative in negative_list]

                    try:
                        # Forward pass
                        anchor_embedding = model(anchor_data)  # Shape: (batch_size, embedding_dim)
                        # Iterate over each sample in the batch

                        positives = [model(positive) for positive in positive_data]
                        negatives = [model(negative) for negative in negative_data]

                        # Compute loss
                        loss = model.info_nce_loss(anchor_embedding, positives, negatives, temperature)

                        # Backward pass and optimization
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # Accumulate loss
                        total_train_loss += loss.item()

                        # Update the batch progress bar with the current loss
                        batch_bar.set_postfix({"Batch Loss": loss.item()})
                    except Exception as e:
                        # If an error occurs, print the error and skip this batch
                        print(f"Skipping batch due to error: {e}")
                        traceback.print_exc()
                        continue
                else:
                    # Anchor-free: (positive_pairs, negative_pairs)
                    positive_pairs, negative_pairs = batch
                    # Move all pairs to the specified device
                    positive_pairs = [(pair[0].to(device), pair[1].to(device)) for pair in positive_pairs]
                    negative_pairs = [(pair[0].to(device), pair[1].to(device)) for pair in negative_pairs]

                    try:
                        # Compute loss for anchor-free case
                        loss = model.info_nce_loss_withoutAnchor(positive_pairs, negative_pairs, temperature)

                        # Backward pass and optimization
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # Accumulate loss
                        total_train_loss += loss.item()

                        # Update the batch progress bar with the current loss
                        batch_bar.set_postfix({"Batch Loss": loss.item()})
                    except Exception as e:
                        print(f"Skipping batch due to error: {e}")
                        continue

            # Calculate average training loss for the epoch
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            # Validation phase
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False, unit="batch")
                for batch in val_bar:
                    if use_anchor:
                        # Anchor-based: (anchor_data, positive_list, negative_list)
                        # Move data to the specified device
                        anchor_data, positive_list, negative_list = batch
                        # Move data to the specified device
                        anchor_data = anchor_data.to(device)
                        positive_data = [positive.to(device) for positive in positive_list]
                        negative_data = [negative.to(device) for negative in negative_list]

                        try:
                            # Forward pass
                            anchor_embedding = model(anchor_data)  # Shape: (batch_size, embedding_dim)
                        # Iterate over each sample in the batch

                            positives = [model(positive) for positive in positive_data]
                            negatives = [model(negative) for negative in negative_data]

                            # Compute loss
                            loss = model.info_nce_loss(anchor_embedding, positives, negatives, temperature)

                            # Accumulate loss
                            total_val_loss += loss.item()

                            # Update the validation progress bar with the current loss
                            val_bar.set_postfix({"Validation Loss": loss.item()})
                        except Exception as e:
                            # If an error occurs, print the error and skip this batch
                            print(f"Skipping batch due to error: {e}")
                            continue

                    else:
                        # Anchor-free: (positive_pairs, negative_pairs)
                        positive_pairs, negative_pairs = batch
                        # Move all pairs to the specified device
                        positive_pairs = [(pair[0].to(device), pair[1].to(device)) for pair in positive_pairs]
                        negative_pairs = [(pair[0].to(device), pair[1].to(device)) for pair in negative_pairs]

                        try:
                            # Compute loss for anchor-free case
                            loss = model.info_nce_loss_withoutAnchor(positive_pairs, negative_pairs, temperature)

                            # Accumulate loss
                            total_val_loss += loss.item()

                            # Update the validation progress bar with the current loss
                            val_bar.set_postfix({"Validation Loss": loss.item()})
                        except Exception as e:
                            print(f"Skipping batch due to error: {e}")
                            continue

            # Calculate average validation loss for the epoch
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Update the main epoch progress bar with losses
            tqdm.write(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
            )

            # Optionally, save the model checkpoint if it's the best so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                # Save the best model
                if hasattr(self, 'save_path') and self.save_path:
                    os.makedirs(self.save_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))
                    tqdm.write(f"Saved best model at epoch {epoch} with Val Loss: {avg_val_loss:.4f}")
            else:
                epochs_without_improvement += 1
                tqdm.write(f"Epochs without improvement: {epochs_without_improvement}/{patience}")

            # Early stopping check
            if epochs_without_improvement >= patience:
                tqdm.write(f"Early stopping triggered after {epoch} epochs.")
                break

        return avg_train_loss, train_losses, val_losses


    def bayes_optimizer(self, dataset, project_path):
        """
        Perform Bayesian Optimization to find the best hyperparameters for the GNN model,
        handling both anchor-based and anchor-free contrastive learning.

        Args:
            dataset: The ContrastiveLearningDataset instance.
            project_path: Path to save the best model and loss plots.
            use_anchor: Boolean flag to indicate whether to use anchor-based or anchor-free contrastive learning.

        Returns:
            optimizer: The BayesianOptimization object after optimization.
        """
        from bayes_opt import BayesianOptimization
        use_anchor = self.is_anchor_based

        # Define the parameter bounds for Bayesian optimization
        pbounds = {
            'hidden_dim': (32, 256),         # Range for the number of hidden dimensions
            'learning_rate': (1e-6, 1e-3),  # Learning rate range
            'dropout_rate': (0.0, 0.3),     # Dropout range
            'weight_decay': (1e-6, 1e-4),   # Weight decay range
        }

        best_train_losses = None
        best_val_losses = None
        best_model_state = None
        best_hyperparams = None
        best_val_loss = float('inf')  # Initialize the best validation loss to infinity

        optimization_results = []  # List to store optimization results

        # Define the function to optimize
        def train_and_evaluate(hidden_dim, learning_rate, weight_decay, dropout_rate):
            nonlocal best_train_losses, best_val_losses, best_model_state, best_val_loss, best_hyperparams

            # Cast hyperparameters to correct types
            hidden_dim = int(hidden_dim)
            learning_rate = float(learning_rate)
            dropout_rate = float(dropout_rate)
            weight_decay = float(weight_decay)

            # Instantiate the model with the current hyperparameters
            model = GNNModelWithContrastiveLearning(
                num_node_features=self.num_node_features,
                num_edge_features=self.num_edge_features,
                num_global_features=self.num_global_features,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                device=self.device,
                is_anchor_based=use_anchor
            ).to(self.device)

            # Train the model and get losses
            num_epochs = 100  # Number of epochs for training
            train_loss, train_losses, val_losses = model.train_model(
                dataset=dataset,
                num_epochs=num_epochs,
                lr=learning_rate,
                weight_decay=weight_decay,
                temperature=0.1,
                device=self.device,
                batch_size=self.batch_size
            )

            current_val_loss = val_losses[-1]  # Get the validation loss for the last epoch

            # Save the best model state and hyperparameters if current validation loss is lower
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_train_losses = train_losses
                best_val_losses = val_losses
                best_model_state = model.state_dict()
                best_hyperparams = {
                    'hidden_dim': hidden_dim,
                    'learning_rate': learning_rate,
                    'dropout_rate': dropout_rate,
                    'weight_decay': weight_decay
                }

            optimization_results.append({
                'hidden_dim': hidden_dim,
                'learning_rate': learning_rate,
                'dropout_rate': dropout_rate,
                'weight_decay': weight_decay,
                'train_loss': train_loss,
                'val_loss': current_val_loss
            })

            return -current_val_loss  # Bayesian optimization maximizes, so return negative val loss

        # Initialize Bayesian optimizer
        optimizer = BayesianOptimization(
            f=train_and_evaluate,  # Function to optimize
            pbounds=pbounds,       # Parameter bounds
            random_state=42,
            verbose=2
        )

        # Run optimization
        optimizer.maximize(init_points=5, n_iter=25)

        print("Best parameters found:", optimizer.max)
        log_dir = os.path.join(project_path, 'bayes_opt_log.txt')
        os.makedirs(project_path, exist_ok=True)

        # Save the optimization results to a CSV file
        optimization_df = pd.DataFrame(optimization_results)
        optimization_df.to_csv(log_dir, index=False)
        print(f"Optimization results saved to {log_dir}")

        # After optimization, reinitialize the model with the best hyperparameters
        best_hidden_dim = int(best_hyperparams['hidden_dim'])
        best_dropout_rate = float(best_hyperparams['dropout_rate'])

        # Create the final model with the best hyperparameters
        best_model = GNNModelWithContrastiveLearning(
        num_node_features=self.num_node_features,
        num_edge_features=self.num_edge_features,
        num_global_features=self.num_global_features,
        hidden_dim=best_hidden_dim,
        dropout_rate=best_dropout_rate
        ).to(self.device)

        # Load the best model state
        best_model.load_state_dict(best_model_state)

        # Save the best model
        model_dir = os.path.join(project_path, 'gnn_model')
        figure_dir = os.path.join(project_path, 'figure')

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(figure_dir, exist_ok=True)

        # Save the best model
        torch.save(best_model.state_dict(), os.path.join(model_dir, 'cl_model_best.pth'))
        print("Best model saved to", os.path.join(model_dir, 'cl_model_best.pth'))

        # Plot the training and validation loss curves
        train_losses = torch.tensor(best_train_losses).cpu().numpy()
        val_losses = torch.tensor(best_val_losses).cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves (Best Configuration)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figure_dir, 'CL_loss_best.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("Final Training Loss:", train_losses[-1])
        print("Final Validation Loss:", val_losses[-1])

        return optimizer
        

