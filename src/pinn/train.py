import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os

from .model import navier_stokes_loss

def generate_fluid_data(n_samples, t_range, x_range, y_range, include_temp=True):
    t = np.random.uniform(*t_range, n_samples)
    x = np.random.uniform(*x_range, n_samples)
    y = np.random.uniform(*y_range, n_samples)
    
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(-t)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.exp(-t)
    p = 0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)) * np.exp(-2*t)
    
    if include_temp:
        T = np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-t)
        x_tensor = torch.tensor(np.column_stack([t, x, y, T]), dtype=torch.float32)
        y_tensor = torch.tensor(np.column_stack([u, v, p, T]), dtype=torch.float32)
    else:
        x_tensor = torch.tensor(np.column_stack([t, x, y]), dtype=torch.float32)
        y_tensor = torch.tensor(np.column_stack([u, v, p]), dtype=torch.float32)
    
    return x_tensor, y_tensor

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_navier_stokes_pinn(model, x_train, y_train, x_val, y_val, rho, nu, alpha, epochs=1000, batch_size=32, lr=0.001, patience=50):
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    early_stopping = EarlyStopping(patience=patience)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            loss = navier_stokes_loss(model, x_batch, y_batch, rho, nu, alpha)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = navier_stokes_loss(model, x_val, y_val, rho, nu, alpha).item()
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        early_stopping(val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_history.png')
    plt.close()
    
    return model, train_losses, val_losses

def save_model(model, path='models'):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'navier_stokes_pinn.pth'))
    print(f"Model saved to {os.path.join(path, 'navier_stokes_pinn.pth')}")

def load_model(model, path='models'):
    model.load_state_dict(torch.load(os.path.join(path, 'navier_stokes_pinn.pth'), weights_only=True))
    return model
