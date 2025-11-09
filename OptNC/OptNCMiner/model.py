import logging
import pandas as pd
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class OptNCMiner(nn.Module):
    def __init__(self, input_dim=1024, head_dims=[2048], dropout=0.1):
        super().__init__()
        self.input_params = {'input_dim': input_dim, 'head_dims': head_dims, 'dropout': dropout}
        self.heads = nn.ModuleList()
        
        in_dim = input_dim
        for dim in head_dims:
            self.heads.append(nn.Sequential(
                nn.Linear(in_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_dim = dim
            
        self.cos = nn.CosineSimilarity(dim=1)
        self.support_pos = None
        self.support_neg = None

    def forward(self, x):
        left, right = t.split(x, x.shape[1]//2, dim=1)
        for head in self.heads:
            left = head(left)
            right = head(right)
        return t.clamp(self.cos(left, right).reshape(-1, 1), 0, 1)

    def fit(self, x, y, valset=None, support=None, batch_size=128, epochs=500, lr=0.0001, es_thresh=50):
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        x_tensor = t.tensor(x, dtype=t.float32)
        y_tensor = t.tensor(y, dtype=t.float32)
        
        if support is not None:
            if isinstance(support, dict):
                self.support_pos = support
            elif isinstance(support, list):
                self.support_pos, self.support_neg = support

        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = pd.DataFrame(columns=['total', 'validation'])
        best_loss = float('inf')
        patience = 0
        
        for epoch in range(epochs):
            self.train()
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
            
            self.eval()
            with t.no_grad():
                train_loss = criterion(self(x_tensor), y_tensor).item()
                losses.loc[epoch+1, 'total'] = train_loss
                
                if valset is not None:
                    x_val, y_val = valset
                    val_tensor = t.tensor(x_val, dtype=t.float32)
                    val_loss = criterion(self(val_tensor), t.tensor(y_val, dtype=t.float32)).item()
                    losses.loc[epoch+1, 'validation'] = val_loss
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience = 0
                        best_state = self.state_dict()
                    else:
                        patience += 1
                        
                    if patience >= es_thresh:
                        self.load_state_dict(best_state)
                        logging.info(f'Early stopping at epoch {epoch+1}')
                        break
                
                if epoch % 50 == 0:
                    log_msg = f'Epoch {epoch+1}: train_loss={train_loss:.4f}'
                    if valset is not None:
                        log_msg += f', val_loss={val_loss:.4f}'
                    logging.info(log_msg)
        
        return losses, optimizer

    def predict(self, x):
        self.eval()
        with t.no_grad():
            return self(t.tensor(x, dtype=t.float32)).numpy()

    def predict2(self, left, right):
        self.eval()
        with t.no_grad():
            combined = t.cat([t.tensor(left, dtype=t.float32), t.tensor(right, dtype=t.float32)], dim=1)
            return self(combined).numpy()

def save_model(model, filename):
    t.save({
        'model_state_dict': model.state_dict(),
        'support_pos': model.support_pos,
        'support_neg': model.support_neg,
        'params': model.input_params
    }, filename)

def load_model(filename):
    checkpoint = t.load(filename)
    model = OptNCMiner(**checkpoint['params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.support_pos = checkpoint['support_pos']
    model.support_neg = checkpoint['support_neg']
    return model