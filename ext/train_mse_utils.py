import time
import random
from tqdm.autonotebook import tqdm  # from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn.functional as F


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, train_loader, optimizer, device):
    model.train()
    loss_all = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        y_pred = model(batch)
        loss = F.mse_loss(y_pred.squeeze(), batch.y)
        loss.backward()
        loss_all += loss.item() * batch.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def eval(model, loader, device):
    model.eval()
    loss_all = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
          y_pred = model(batch)
          loss = F.mse_loss(y_pred.squeeze(), batch.y)
          loss_all += loss.item() * batch.num_graphs
    return loss_all / len(loader.dataset)

def _run_experiment(model, train_loader, val_loader, test_loader, n_epochs=100, verbose=True, device='cpu'):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.9, patience=25, min_lr=0.00001)
    
    if verbose:
        print(f"Running experiment for {type(model).__name__}.")
        # print("\nModel architecture:")
        # print(model)
        print(f'Total parameters: {total_param}')
        print("\nStart training:")
    
    best_val_loss = None
    perf_per_epoch = [] # Track Test/Val performace vs. epoch (for plotting)
    t = time.time()
    for epoch in range(1, n_epochs+1):
        # Train model for one epoch, return avg. training loss
        loss = train(model, train_loader, optimizer, device)
        
        # Evaluate model on validation set
        val_loss = eval(model, val_loader, device)
        
        if best_val_loss is None or val_loss <= best_val_loss:
            # Evaluate model on test set if validation metric improves
            test_loss = eval(model, test_loader, device)
            best_val_loss = val_loss

        if epoch % 10 == 0 and verbose:
            print(f'Epoch: {epoch:03d}, LR: {lr:.5f}, Loss: {loss:.5f}, '
                  f'Val Acc: {val_loss:.3f}, Test loss: {test_loss:.3f}')
        
        perf_per_epoch.append((test_loss, val_loss, epoch, type(model).__name__))
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
    
    t = time.time() - t
    train_time = t
    if verbose:
        print(f"\nDone! Training took {train_time:.2f}s. Best validation loss: {best_val_loss:.4f}, corresponding test loss: {test_loss:.4f}.")
    
    return best_val_loss, test_loss, train_time, perf_per_epoch


def run_experiment(model, train_loader, val_loader, test_loader, n_epochs=100, n_times=100, verbose=False, device='cpu'):
    print(f"Running experiment for {type(model).__name__} ({device}).")
    
    best_val_loss_list = []
    test_loss_list = []
    train_time_list = []
    for idx in tqdm(range(n_times)):
        seed(idx) # set random seed
        best_val_loss, test_loss, train_time, _ = _run_experiment(model, train_loader, val_loader, test_loader, n_epochs, verbose, device)
        best_val_loss_list.append(best_val_loss)
        test_loss_list.append(test_loss)
        train_time_list.append(train_time)
    
    print(f'\nDone! Averaged over {n_times} runs: \n '
          f'- Training time: {np.mean(train_time_list):.2f}s ± {np.std(train_time_list):.2f}. \n '
          f'- Best validation loss: {np.mean(best_val_loss_list):.4f} ± {np.std(best_val_loss_list):.4f}. \n'
          f'- Test loss: {np.mean(test_loss_list):.4f} ± {np.std(test_loss_list):.4f}. \n')
    
    return best_val_loss_list, test_loss_list, train_time_list

