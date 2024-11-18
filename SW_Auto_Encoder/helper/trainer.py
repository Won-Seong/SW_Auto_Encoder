import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm

class Trainer():
    def __init__(self,
                 model: nn.Module,
                 loss_fn: Callable,
                 optimizer: torch.optim.Optimizer = None,
                 device: torch.device = torch.device("cpu"),
                 no_label : bool = True):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)
        self.loss_fn = loss_fn
        self.device = device
        self.no_label = no_label
        
    def train(self, dl : DataLoader, epochs : int, file_name : str):
        self.model.train()
        best_loss = float("inf")
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            
            for _, batch in enumerate(tqdm(dl, leave=False, desc=f"Epoch {epoch}/{epochs}", colour="#005500")):
                
                if self.no_label: x = batch[0].to(self.device)
                else: x, y = batch[0].to(self.device), batch[1].to(self.device)
                
                x_hat = self.model(x)
                loss = self.loss_fn(x, x_hat)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            log_string = f"Loss at epoch {epoch}: {epoch_loss:.3f}"

            # Storing the model
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                torch.save(self.model.state_dict(), f = 'models/'+ file_name + '_epoch' + str(epoch) + '.pt' )
                log_string += " --> Best model ever (stored)"
            print(log_string)
                
    def evaluate(self, dataloader : DataLoader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if self.no_label: x = batch[0].to(self.device)
                else: x, y = batch[0].to(self.device), batch[1].to(self.device)
            
                x_hat = self.model(x)
                loss = self.loss_fn(x, x_hat)
                total_loss += loss.item()
        
        final_loss = total_loss / len(dataloader)
        print("Loss = " , final_loss)
        return final_loss
        