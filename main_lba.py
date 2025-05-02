import os
import argparse
import numpy as np
from collections import defaultdict
import torch
from functools import partial
import time
from datetime import datetime
import sys
from atom3d.util import metrics
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import sklearn.metrics as sk_metrics
import wandb
sys.path.append('./dataset')
from lbadataset import LBADataset
sys.path.append('./models')
from pronetCHA import ProNet
from gvpgnnCHA import GVPNet

parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', metavar='N', type=int, default=4, help='number of threads for loading data, default=4')
parser.add_argument('--dataset', type=str, default='lba')
parser.add_argument('--data_path', type=str, default='/root/workspace/A_data/split-by-sequence-identity-30/data/')
parser.add_argument('--lba_split', metavar='SPLIT', type=int, choices=[30, 60], help='identity cutoff for LBA, 30 (default) or 60', default=30)

# Training
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lr_decay_step_size', type=int, default=50, help='Learning rate step size')
parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='Learning rate factor') 
parser.add_argument('--weight_decay', type=float, default=0, help='Weight Decay')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size during training')
parser.add_argument('--batch_size_eval', type=int, default=32, help='Batch size during training')
# Model
parser.add_argument('--model', type=str, default='ProNet', help='Choose from \'ProNet\'GVPNet\'')
parser.add_argument('--level', type=str, default='backbone', help='Choose from \'aminoacid\', \'backbone\', and \'allatom\' levels')
parser.add_argument('--num_blocks', type=int, default=3, help='Model layers')
parser.add_argument('--hidden_channels', type=int, default=256, help='Hidden dimension')
parser.add_argument('--out_channels', type=int, default=1, help='Number of classes, 1195 for the fold data, 384 for the ECdata')
parser.add_argument('--fix_dist', action='store_true')  
parser.add_argument('--cutoff', type=float, default=10, help='Distance constraint for building the protein graph') 
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')
parser.add_argument('--schull', type=eval, default=True, help='True | False')
## data augmentation tricks
parser.add_argument('--mask', action='store_true', help='Random mask some node type')
parser.add_argument('--noise', action='store_true', help='Add Gaussian noise to node coords')
parser.add_argument('--deform', action='store_true', help='Deform node coords')
parser.add_argument('--data_augment_eachlayer', action='store_true', help='Add Gaussian noise to features')
parser.add_argument('--euler_noise', action='store_true', help='Add Gaussian noise Euler angles')
parser.add_argument('--mask_aatype', type=float, default=0.1, help='Random mask aatype to 25(unknown:X) ratio')
parser.add_argument('--metric', type=str, default='rmse', help='Choose from \'rmse\', \'pearson\', \'kendall\', and \'spearman\'')
# wandb
parser.add_argument('--wandb', type=str, default='disabled', help='wandb mode')
args = parser.parse_args()

models_dir = '/root/workspace/A_data/DIPS-split/models/dlb'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = torch.nn.MSELoss()

def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Ensure the tensors are 1D and have the same shape
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("Input tensors must be 1D.")
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape.")
    
    # Compute the mean of x and y
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    
    # Compute the covariance
    cov = torch.sum((x - x_mean) * (y - y_mean))
    
    # Compute the standard deviations
    x_std = torch.sqrt(torch.sum((x - x_mean)**2))
    y_std = torch.sqrt(torch.sum((y - y_mean)**2))
    
    # Compute Pearson correlation coefficient
    return cov / (x_std * y_std)

def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Ensure the tensors are 1D and have the same shape
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("Input tensors must be 1D.")
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape.")

    # Rank the elements of the tensors
    x_rank = torch.argsort(torch.argsort(x))
    y_rank = torch.argsort(torch.argsort(y))
    
    # Convert ranks to float tensors
    x_rank = x_rank.float()
    y_rank = y_rank.float()

    # Calculate Pearson correlation on ranks
    return pearson_correlation(x_rank, y_rank)

def kendall_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Ensure the tensors are 1D and have the same shape
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("Input tensors must be 1D.")
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape.")
    
    # Number of concordant and discordant pairs
    n_concordant = 0
    n_discordant = 0

    # Compare all pairs
    n = x.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            sign_x = torch.sign(x[i] - x[j])
            sign_y = torch.sign(y[i] - y[j])
            product = sign_x * sign_y

            if product > 0:
                n_concordant += 1
            elif product < 0:
                n_discordant += 1

    # Calculate Kendall's Tau
    tau = (n_concordant - n_discordant) / (0.5 * n * (n - 1))
    return tau

def get_datasets(data_path= None):
    trainset = LBADataset(data_path+'train', edge_cutoff=args.cutoff)
    valset = LBADataset(data_path+'val')
    testset = LBADataset(data_path+'test')
    return trainset, valset, testset

def get_model(args):
    if args.model == 'ProNet':
        model = ProNet(num_blocks=args.num_blocks, 
                       hidden_channels=args.hidden_channels, 
                       out_channels=args.out_channels,
                       cutoff=args.cutoff, dropout=args.dropout,
                       data_augment_eachlayer=args.data_augment_eachlayer,
                       euler_noise = args.euler_noise, level=args.level, 
                       schull=args.schull).to(device)
    elif args.model == 'GVPNet':
        model = GVPNet(schull=args.schull).to(device)
    return model

def get_metrics():
    def _correlation(metric, targets, predict, ids=None, glob=True):
        if glob: return metric(targets, predict)
        _targets, _predict = defaultdict(list), defaultdict(list)
        for _t, _p, _id in zip(targets, predict, ids):
            _targets[_id].append(_t)
            _predict[_id].append(_p)
        return np.mean([metric(_targets[_id], _predict[_id]) for _id in _targets])
    
    correlations = {
        'pearson': partial(_correlation, metrics.pearson),
        'kendall': partial(_correlation, metrics.kendall),
        'spearman': partial(_correlation, metrics.spearman)
    }
    return {**correlations, 'rmse': partial(sk_metrics.mean_squared_error, squared=False)}



def train(args, model, loader, optimizer, device):
    model.train()
    train_loss = 0
    train_num = 0
    for _, batch in enumerate(tqdm(loader, disable=False)):
        if args.mask:
            # random mask node aatype
            mask_indice = torch.tensor(np.random.choice(batch.num_nodes, int(batch.num_nodes * args.mask_aatype), replace=False))
            batch.x[:, 0][mask_indice] = 25
        if args.noise:
            # add gaussian noise to atom coords
            gaussian_noise = torch.clip(torch.normal(mean=0.0, std=0.1, size=batch.coords_ca.shape), min=-0.3, max=0.3)
            batch.coords_ca += gaussian_noise
            if args.level != 'aminoacid':
                batch.coords_n += gaussian_noise
                batch.coords_c += gaussian_noise
        if args.deform:
            # Anisotropic scale
            deform = torch.clip(torch.normal(mean=1.0, std=0.1, size=(1, 3)), min=0.9, max=1.1)
            batch.coords_ca *= deform
            if args.level != 'aminoacid':
                batch.coords_n *= deform
                batch.coords_c *= deform
        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(batch).squeeze(dim=-1)
        label = batch.label
        if args.metric == 'rmse':
            batch_loss = criterion(pred, label)
        elif args.metric == 'pearson':
            batch_loss = -pearson_correlation(pred, label)
        elif args.metric == 'spearman':
            batch_loss = -spearman_correlation(pred, label)
        elif args.metric == 'kendall':
            batch_loss = -kendall_correlation(pred, label)
        else:
            raise ValueError('Invalid metric')
        batch_loss = criterion(pred, label)
        batch_loss.backward()
        optimizer.step()

        train_loss += batch_loss.item() * batch.label.shape[0]
        train_num += batch.label.shape[0]
        
    return train_loss / train_num

def val(model, loader, device):
    model.eval()
    metrics = get_metrics()
    targets, predicts = [], []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader, disable=False)):
            batch = batch.to(device)
            pred = model(batch).squeeze(dim=-1)
            label = batch.label
            targets.extend(list(label.cpu().numpy()))
            predicts.extend(list(pred.cpu().numpy()))
    val_dict = {} 
    for name, func in metrics.items():
        value = func(targets, predicts)
        val_dict[name] = value
    return val_dict

def test(model, loader, device):
    model.eval()
    metrics = get_metrics()
    targets, predicts = [], []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader, disable=False)):
            batch = batch.to(device)
            pred = model(batch).squeeze(dim=-1)
            label = batch.label
            targets.extend(list(label.cpu().numpy()))
            predicts.extend(list(pred.cpu().numpy()))
    test_dict = {} 
    for name, func in metrics.items():
        value = func(targets, predicts)
        test_dict[name] = value
    return test_dict

def main():

    save_dir = '/root/workspace/A_out/ProteinSCHull/trained_models_{dataset}_{model}/{level}/layer{num_blocks}_cutoff{cutoff}_hidden{hidden_channels}_batch{batch_size}_lr{lr}_{lr_decay_factor}_{lr_decay_step_size}_dropout{dropout}__{time}'.format(
                dataset=args.dataset, model=args.model, level=args.level, 
                num_blocks=args.num_blocks, cutoff=args.cutoff, hidden_channels=args.hidden_channels, batch_size=args.batch_size, 
                lr=args.lr, lr_decay_factor=args.lr_decay_factor, 
                lr_decay_step_size=args.lr_decay_step_size, dropout=args.dropout, time=datetime.now())
    if not save_dir == "" and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    proj_name = 'trained_models_{model}/{level}/schull{schull}_layer{num_blocks}_cutoff{cutoff}_hidden{hidden_channels}_batch{batch_size}_lr{lr}_{lr_decay_factor}_{lr_decay_step_size}_dropout{dropout}__{time}'.format(
                 model=args.model, level=args.level, schull=args.schull,
                 num_blocks=args.num_blocks, cutoff=args.cutoff, hidden_channels=args.hidden_channels, batch_size=args.batch_size, 
                 lr=args.lr, lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size, dropout=args.dropout, time=datetime.now())
    wandb.init(entity='utah-math-data-science', 
           project='SCHull_on_LBA_02', 
           mode=args.wandb,
           name=proj_name, 
           dir='/root/workspace/A_data/split-by-sequence-identity-30/',
           config=args)
    
    # data and data loaders
    trainset, valset, testset = get_datasets(args.data_path)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(valset, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(testset, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers)
    # model and optimizer
    model = get_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                 weight_decay=args.weight_decay) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=args.lr_decay_step_size, 
                                                gamma=args.lr_decay_factor)
    num_params = sum(p.numel() for p in model.parameters()) 
    print('num_parameters:', num_params)
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        t_start = time.perf_counter()
        train_loss = train(args, model, train_loader, optimizer, device)
        t_end_train = time.perf_counter()
        val_dict = val(model, val_loader, device)
        val_loss = -val_dict[args.metric] if args.metric in ['pearson', 'spearman', 'kendall'] else val_dict[args.metric]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir+'_best.pth')
            test_results = test(model, test_loader, device)
            test_result_at_best_val = test_results[args.metric]

        print('Epoch: {} | Train Loss: {:.6g} | Val Loss {:.6g} | Training Time: {:.4g}'.format(epoch, train_loss, val_loss, t_end_train - t_start))
        wandb.log({'epoch': epoch, 
                   'train_loss': train_loss, 
                   'val_loss': val_loss, 
                   'best_val_loss': best_val_loss,
                   'test_{}_at_best_val'.format(args.metric): test_result_at_best_val, })

        scheduler.step() 


if __name__ == '__main__':
    main()