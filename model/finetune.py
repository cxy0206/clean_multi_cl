import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from itertools import product
import math
import warnings
warnings.filterwarnings("ignore")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fixed hyperparameters
LEARNING_RATE = 1e-4
EMB_DIM = 512
PATIENCE_ES = 20  # early stopping patience
PATIENCE_LR = 10   # scheduler patience
LR_FACTOR = 0.5   # scheduler reduce factor

# User modules
from model.featurisation import smiles2graph
from model.CL_model_vas_info import GNNModelWithNewLoss
from model.fusion import TransformerFusionModel, WeightedFusion, MLP, FusionFineTuneModel

# Utility: hyperparameter grid
def product_dict(grid):
    for combo in product(*grid.values()):
        yield dict(zip(grid.keys(), combo))

# Data loading as in your Jupyter snippet
def load_data(name, batch_size=32, val_split=0.1, test_split=0.2, seed=42):
    df = pd.read_csv(f'data/{name}.csv')
    smiles_list = df['smiles'].tolist()
    labels = df[name].tolist()
    data_list = smiles2graph(smiles_list, labels)
    train_val, test_data = train_test_split(data_list, test_size=test_split, random_state=seed)
    train_data, val_data = train_test_split(
        train_val, test_size=val_split/(1 - test_split), random_state=seed
    )
    train_loader = GeoDataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = GeoDataLoader(val_data,   batch_size=batch_size, shuffle=False)
    test_loader  = GeoDataLoader(test_data,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Load pre-trained encoders
def load_pretrained_encoders(sample):
    encoders = []
    for i in range(3):
        enc = GNNModelWithNewLoss(
            num_node_features=sample.x.shape[1],
            num_edge_features=sample.edge_attr.shape[1],
            num_global_features=sample.global_features.shape[0],
            hidden_dim=EMB_DIM
        ).to(device)
        ckpt = torch.load(f'premodels/{i}/best_model.pth', map_location=device)
        enc.load_state_dict(ckpt['encoder_state_dict'])
        encoders.append(enc)
    return encoders

# Build fusion model
def get_finetune_model(fusion_method, sample, dropout):
    encoders = load_pretrained_encoders(sample)
    if fusion_method == 'attention':
        fusion = TransformerFusionModel(emb_dim=EMB_DIM).to(device)
    elif fusion_method == 'weighted':
        fusion = WeightedFusion(num_inputs=3, emb_dim=EMB_DIM, dropout=dropout).to(device)
    elif fusion_method == 'concat':
        fusion = MLP(emb_dim=EMB_DIM * 3).to(device)
    else:
        raise ValueError(f'Unknown fusion method {fusion_method}')
    return FusionFineTuneModel(encoders, fusion, fusion_method).to(device)

# Single training routine
# Training loop
def train_and_validate(model, train_loader, val_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=PATIENCE_LR)

    train_losses, val_rmses = [], []
    best_val_rmse = float('inf')
    best_state = None
    patience_cnt = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            embs = [encoder(batch) for encoder in model.encoders]  # List of [B, D]
            embs = torch.stack(embs, dim=1)  # [B, 3, D]

            # Handle fusion method
            if model.fusion_method == 'concat':
                # Flatten for MLP
                embs = embs.view(embs.size(0), -1)  # [B, 3 * D]
            
            out = model.fusion(embs)  # Fusion output
            pred = out[0] if isinstance(out, tuple) else out  # Ensure proper unpacking
            label = batch.y.view(-1).float().to(device)
            loss = criterion(pred, label).sqrt()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            preds, labs = [], []
            for batch in val_loader:
                batch = batch.to(device)
                embs = [encoder(batch) for encoder in model.encoders]  # List of [B, D]
                embs = torch.stack(embs, dim=1)  # [B, 3, D]

                if model.fusion_method == 'concat':
                    # Flatten for MLP
                    embs = embs.view(embs.size(0), -1)  # [B, 3 * D]
                
                out = model.fusion(embs)  # Fusion output
                pred = out[0] if isinstance(out, tuple) else out  # Ensure proper unpacking
                preds.append(pred.cpu())
                labs.append(batch.y.view(-1).cpu())
            preds = torch.cat(preds)
            labs = torch.cat(labs)
            rmse = criterion(preds, labs).sqrt().item()
        val_rmses.append(rmse)
        scheduler.step(rmse)

        # Early stopping
        if rmse < best_val_rmse - 1e-6:
            best_val_rmse = rmse
            best_state = model.state_dict().copy()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE_ES:
                break

        print(f"[Epoch {epoch:03d}] Train Loss={avg_loss:.4f}, Val RMSE={rmse:.4f}")

    # Load best
    model.load_state_dict(best_state)
    return train_losses, val_rmses


# Final test evaluation
def test_model(model, test_loader):
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        preds, labs = [], []
        for batch in test_loader:
            batch = batch.to(device)
            embs = [encoder(batch) for encoder in model.encoders]  # List of [B, D]
            embs = torch.stack(embs, dim=1)  # [B, 3, D]

            if model.fusion_method == 'concat':
                # Flatten for MLP
                embs = embs.view(embs.size(0), -1)  # [B, 3 * D]

            out = model.fusion(embs)  # Fusion output
            pred = out if not isinstance(out, tuple) else out[0]  # Handle single or tuple output
            preds.append(pred.cpu())
            labs.append(batch.y.view(-1).cpu())
        preds = torch.cat(preds)
        labs = torch.cat(labs)
        rmse = criterion(preds, labs).sqrt().item()
    return rmse, preds.numpy(), labs.numpy()

# Hyperparameter search
def hyperparam_search(fusion_method, name, grid):
    best = {'params': None, 'rmse': float('inf')}
    for params in product_dict(grid):
        tr, vl, _ = load_data(name,
            batch_size=params['batch_size'], val_split=params['val_split'],
            test_split=params['test_split'], seed=params['seed'])
        sample = tr.dataset[0]
        model = get_finetune_model(fusion_method, sample, dropout=params['dropout'])
        train_losses, val_rmses = train_and_validate(model, tr, vl, epochs=params['epochs'])
        if val_rmses[-1] < best['rmse']:
            best = {'params': params, 'rmse': val_rmses[-1]}
    return best

# Multi-run for each best config
def run_multiple(fusion_method, name, params, runs=10):
    rmses, histories, results = [], [], []
    for i in range(runs):
        tr, vl, te = load_data(name,
            batch_size=params['batch_size'], val_split=params['val_split'],
            test_split=params['test_split'], seed=42+i)
        sample = tr.dataset[0]
        model = get_finetune_model(fusion_method, sample, dropout=params['dropout'])
        tr_losses, val_rmses = train_and_validate(model, tr, vl, epochs=params['epochs'])
        rmses.append(val_rmses[-1])
        histories.append((tr_losses, val_rmses))
        rmse_test, preds, labs = test_model(model, te)
        results.append((preds, labs))
    idx = np.argsort(rmses)[:3]
    mean_top3 = np.mean([rmses[i] for i in idx])
    var_top3  = np.var ([rmses[i] for i in idx])
    best_idx = idx[0]
    best_history = histories[best_idx]
    best_preds, best_labs = results[best_idx]
    return {
        'mean_top3': mean_top3,
        'var_top3': var_top3,
        'best_history': best_history,
        'best_preds': best_preds,
        'best_labels': best_labs
    }

# Plot helpers
def plot_training(train_losses, val_rmses, ds, method, out_dir):
    plt.figure(); plt.plot(train_losses, label='Train Loss'); plt.plot(val_rmses, label='Val RMSE')
    plt.xlabel('Epoch'); plt.ylabel('Metric'); plt.legend(); plt.title(f'{ds}-{method}')
    plt.savefig(f'{out_dir}/{ds}_{method}_train.png'); plt.close()

def plot_test_distribution(preds, labs, ds, method, out_dir):
    plt.figure(); plt.scatter(labs, preds, alpha=0.5)
    mn, mx = labs.min(), labs.max()
    plt.plot([mn, mx], [mn, mx], 'k--'); plt.xlabel('True'); plt.ylabel('Pred')
    plt.title(f'{ds}-{method} Test'); plt.savefig(f'{out_dir}/{ds}_{method}_test.png'); plt.close()

# Main execution
if __name__ == '__main__':
    datasets = ['freesolv', 'lipo', 'esol']
    fusion_methods = ['concat', 'weighted', 'attention']
    param_grid = {
        'weight_decay': [0, 1e-5],
        'dropout': [0.1, 0.3],
        'batch_size': [16, 64],
        'epochs': [100, 200],
        'val_split': [0.1],
        'test_split': [0.2],
        'seed': [42]
    }
    os.makedirs('results', exist_ok=True)
    summary = []
    for ds in datasets:
        for fm in fusion_methods:
            print(f'Grid search {ds}-{fm}')
            best = hyperparam_search(fm, ds, param_grid)
            print('Best:', best)
            res = run_multiple(fm, ds, best['params'], runs=3)
            plot_training(*res['best_history'], ds, fm, 'results')
            plot_test_distribution(res['best_preds'], res['best_labels'], ds, fm, 'results')
            summary.append({
                'dataset': ds, 'fusion': fm, 'params': best['params'],
                'mean_top3': res['mean_top3'], 'var_top3': res['var_top3']
            })
    pd.DataFrame(summary).to_csv('results/summary.csv', index=False)
