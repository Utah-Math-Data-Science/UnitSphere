import os
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import argparse

import torch
import torch.optim as optim
from torch import nn 
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('/root/workspace/programme/SCHull/models')
from pronetCHA import ProNet
from gvpgnnCHA import GVPNet
from segnn import SEGNN
from mace import MACE
sys.path.append('/root/workspace/programme/SCHull/dataset')
from ECdataset import ECdataset
from FDdataset import FOLDdataset
from atom3d.datasets import LMDBDataset
from torch_geometric.data import DataLoader

import wandb
import warnings
warnings.filterwarnings("ignore")

criterion = nn.CrossEntropyLoss()

num_fold = 1195
num_func = 384


def train(args, model, loader, optimizer, device):
    model.train()

    loss_accum = 0
    preds = []
    functions = []
    for step, batch in enumerate(tqdm(loader, disable=args.disable_tqdm)):
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
                     
        try:
            pred = model(batch) 
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): 
                print('\n forward error \n')
                raise(e)
            else:
                print('OOM')
            torch.cuda.empty_cache()
            continue
        preds.append(torch.argmax(pred, dim=1))        
        function = batch.y
        functions.append(function)
        optimizer.zero_grad()
        loss = criterion(pred, function)
        loss.backward()
        optimizer.step()

        loss_accum += loss.item()        

    functions = torch.cat(functions, dim=0)
    preds = torch.cat(preds, dim=0)
    acc = torch.sum(preds==functions)/functions.shape[0]
    
    return loss_accum/(step + 1), acc.item()


def evaluation(args, model, loader, device):    
    model.eval()
    
    loss_accum = 0
    preds = []
    functions = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        # pred = model(batch)
        try:
            pred = model(batch) 
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): 
                print('\n forward error \n')
                raise(e)
            else:
                print('evaluation OOM')
            torch.cuda.empty_cache()
            continue
        preds.append(torch.argmax(pred, dim=1))
        
        function = batch.y
        functions.append(function)
        loss = criterion(pred, function)
        loss_accum += loss.item()    
            
    functions = torch.cat(functions, dim=0)
    preds = torch.cat(preds, dim=0)
    acc = torch.sum(preds==functions)/functions.shape[0]
    
    return loss_accum/(step + 1), acc.item()

    
def main():
    ### Args
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=9, help='Device to use')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of workers in Dataloader')

    ### Data
    parser.add_argument('--dataset', type=str, default='fold', help='func or fold')
    parser.add_argument('--dataset_path', type=str, default='/root/workspace/A_data', help='path to load and process the data')
    
    # data augmentation tricks
    parser.add_argument('--mask', action='store_true', help='Random mask some node type')
    parser.add_argument('--noise', action='store_true', help='Add Gaussian noise to node coords')
    parser.add_argument('--deform', action='store_true', help='Deform node coords')
    parser.add_argument('--data_augment_eachlayer', action='store_true', help='Add Gaussian noise to features')
    parser.add_argument('--euler_noise', action='store_true', help='Add Gaussian noise Euler angles')
    parser.add_argument('--mask_aatype', type=float, default=0.1, help='Random mask aatype to 25(unknown:X) ratio')
    
    ### Model
    parser.add_argument('--model', type=str, default='ProNet', help='Choose from \'ProNet\'GVPNet\'SEGNN\'\MACE')
    parser.add_argument('--level', type=str, default='backbone', help='Choose from \'aminoacid\', \'backbone\', and \'allatom\' levels')
    parser.add_argument('--num_blocks', type=int, default=3, help='Model layers')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--out_channels', type=int, default=1195, help='Number of classes, 1195 for the fold data, 384 for the ECdata')
    parser.add_argument('--fix_dist', action='store_true')  
    parser.add_argument('--cutoff', type=float, default=8, help='Distance constraint for building the protein graph') 
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')
    parser.add_argument('--schull', type=eval, default=False, help='True | False')

    ### Training hyperparameter
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--lr_decay_step_size', type=int, default=50, help='Learning rate step size')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='Learning rate factor') 
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight Decay')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size during training')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size')
    
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--save_dir', type=str, default=None, help='Trained model path')
    parser.add_argument('--wandb', type=str, default='disabled')
    parser.add_argument('--disable_tqdm', default=False, action='store_true')
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    ##### load datasets
    print('Loading Train & Val & Test Data...')
    if args.dataset == 'func':
        try:
            train_set = ECdataset(root=args.dataset_path + '/ProtFunct', split='Train')
            val_set = ECdataset(root=args.dataset_path + '/ProtFunct', split='Val')
            test_set = ECdataset(root=args.dataset_path + '/ProtFunct', split='Test')
        except FileNotFoundError: 
            print('\n Please download data firstly, following https://github.com/divelab/DIG/tree/dig-stable/dig/threedgraph/dataset#ecdataset-and-folddataset and https://github.com/phermosilla/IEConv_proteins#download-the-preprocessed-datasets \n')
            raise(FileNotFoundError)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        print('Done!')
        print('Train, val, test:', train_set, val_set, test_set)
    elif args.dataset == 'fold':
        test_lst = [200, 2000]
 
        train_set = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='training', n_node=-1, ct_lst=test_lst)
        val_set = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='validation', n_node=-1, ct_lst=test_lst)
        test_set_dict = {}
        for k in range(len(test_lst)):
            test_set_dict['test_fold_{}'.format(int(test_lst[k]))] = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='test_fold'.format(int(test_lst[k])), n_node=k, ct_lst=test_lst)
            test_set_dict['test_super_{}'.format(int(test_lst[k]))] = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='test_superfamily'.format(int(test_lst[k])), n_node=k, ct_lst=test_lst)
            test_set_dict['test_family_{}'.format(int(test_lst[k]))] = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='test_family'.format(int(test_lst[k])), n_node=k, ct_lst=test_lst)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader_dict = {}
        for k in test_lst:
            test_loader_dict['test_fold_{}_loader'.format(int(k))] = DataLoader(test_set_dict['test_fold_{}'.format(int(k))], batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            test_loader_dict['test_super_{}_loader'.format(int(k))] = DataLoader(test_set_dict['test_super_{}'.format(int(k))], batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            test_loader_dict['test_family_{}_loader'.format(int(k))] = DataLoader(test_set_dict['test_family_{}'.format(int(k))], batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        print('Done!')
        # print('Train, val, test (fold, superfamily, family):', train_set, val_set, test_fold, test_super, test_family)
    else:
        print('not supported dataset')
    
    ##### set up model
    if args.model == 'ProNet':
        model = ProNet(num_blocks=args.num_blocks, hidden_channels=args.hidden_channels, out_channels=args.out_channels,
                cutoff=args.cutoff, dropout=args.dropout,
                data_augment_eachlayer=args.data_augment_eachlayer,
                euler_noise = args.euler_noise, level=args.level, schull=args.schull).to(device)
    elif args.model == 'GVPNet':
        model = GVPNet(schull=args.schull).to(device)
    elif args.model == 'SEGNN':
        model = SEGNN(cutoff=args.cutoff, dropout=args.dropout, 
                      in_dim=1, out_dim=1, 
                      hidden_features=args.hidden_channels, 
                      num_layers=args.num_blocks, schull=args.schull).to(device)
    elif args.model == 'MACE':
        model = MACE(r_max=args.cutoff,
                     num_layers=args.num_blocks,
                     mlp_dim=args.hidden_channels, 
                     out_channels=args.out_channels,
                     dropout=args.dropout, 
                     schull=args.schull).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor)
    
    
    if args.continue_training:
        save_dir = args.save_dir
        checkpoint = torch.load(save_dir + '/best_val.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        save_dir = '/root/workspace/A_out/ProteinSCHull/trained_models_{dataset}/{level}/layer{num_blocks}_cutoff{cutoff}_hidden{hidden_channels}_batch{batch_size}_lr{lr}_{lr_decay_factor}_{lr_decay_step_size}_dropout{dropout}__{time}'.format(
            dataset=args.dataset, level=args.level, 
            num_blocks=args.num_blocks, cutoff=args.cutoff, hidden_channels=args.hidden_channels, batch_size=args.batch_size, 
            lr=args.lr, lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size, dropout=args.dropout, time=datetime.now())
        print('saving to...', save_dir)
        start_epoch = 1

    proj_name = 'trained_{model}_{dataset}/{level}/schull{schull}_layer{num_blocks}_cutoff{cutoff}_hidden{hidden_channels}_batch{batch_size}_lr{lr}_{lr_decay_factor}_{lr_decay_step_size}_dropout{dropout}__{time}'.format(
                 model=args.model, dataset=args.dataset, level=args.level, schull=args.schull,
                 num_blocks=args.num_blocks, cutoff=args.cutoff, hidden_channels=args.hidden_channels, batch_size=args.batch_size, 
                 lr=args.lr, lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size, dropout=args.dropout, time=datetime.now())
    wandb.init(entity='utah-math-data-science', 
           project='ProNet_SCHull_3_Protein_segnn', 
           mode=args.wandb,
           name=proj_name, 
           dir='/root/workspace/A_data/HomologyTAPE/',
           config=args)
    
    num_params = sum(p.numel() for p in model.parameters()) 
    print('num_parameters:', num_params)

    if args.dataset == 'func':
        writer = SummaryWriter(log_dir=save_dir)
        best_val_acc = 0
        test_at_best_val_acc = 0
        
        for epoch in range(start_epoch, args.epochs+1):
            print('==== Epoch {} ===='.format(epoch))
            t_start = time.perf_counter()
            
            train_loss, train_acc = train(args, model, train_loader, optimizer, device)
            t_end_train = time.perf_counter()
            val_loss, val_acc = evaluation(args, model, val_loader, device)
            t_start_test = time.perf_counter()
            test_loss, test_acc = evaluation(args, model, test_loader, device)
            t_end_test = time.perf_counter() 

            if not save_dir == "" and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if not save_dir == "" and val_acc > best_val_acc:
                print('Saving best val checkpoint ...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
                torch.save(checkpoint, save_dir + '/best_val.pt')
                best_val_acc = val_acc    
                test_at_best_val_acc = test_acc       

            t_end = time.perf_counter()
            print('Train: Loss:{:.6f} Acc:{:.4f}, Validation: Loss:{:.6f} Acc:{:.4f}, Test: Loss:{:.6f} Acc:{:.4f}, test_acc@best_val:{:.4f}, time:{}, train_time:{}, test_time:{}'.format(
                train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, test_at_best_val_acc, t_end-t_start, t_end_train-t_start, t_end_test-t_start_test))
            
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_acc', val_acc, epoch)
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)
            writer.add_scalar('test_at_best_val_acc', test_at_best_val_acc, epoch)

            scheduler.step()   
        
        writer.close()    
        # Save last model
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
        torch.save(checkpoint, save_dir + "/epoch{}.pt".format(epoch))
    
    elif args.dataset == 'fold':
        writer = SummaryWriter(log_dir=save_dir)
        best_val_acc = 0
        test_loss_dict = {}
        for k in test_lst:
            test_loss_dict['test_fold_loss_{}'.format(k)] = 0
            test_loss_dict['test_super_loss_{}'.format(k)] = 0
            test_loss_dict['test_family_loss_{}'.format(k)] = 0

            test_loss_dict['test_fold_acc_{}'.format(k)] = 0
            test_loss_dict['test_super_acc_{}'.format(k)] = 0
            test_loss_dict['test_family_acc_{}'.format(k)] = 0

            test_loss_dict['test_fold_at_best_val_acc_{}'.format(k)] = 0
            test_loss_dict['test_super_at_best_val_acc_{}'.format(k)] = 0
            test_loss_dict['test_family_at_best_val_acc_{}'.format(k)] = 0

            test_loss_dict['best_test_fold_acc_{}'.format(k)] = 0 
            test_loss_dict['best_test_super_acc_{}'.format(k)] = 0
            test_loss_dict['best_test_family_acc_{}'.format(k)] = 0
        
        for epoch in range(start_epoch, args.epochs+1):
            print('==== Epoch {} ===='.format(epoch))
            t_start = time.perf_counter()
            
            train_loss, train_acc = train(args, model, train_loader, optimizer, device)
            t_end_train = time.perf_counter()
            val_loss, val_acc = evaluation(args, model, val_loader, device)
            
            t_start_test = time.perf_counter()
            for k in test_lst:
                test_loss_dict['test_fold_loss_{}'.format(k)], test_loss_dict['test_fold_acc_{}'.format(k)] = evaluation(args, model, test_loader_dict['test_fold_{}_loader'.format(int(k))], device)
                test_loss_dict['test_super_loss_{}'.format(k)], test_loss_dict['test_super_acc_{}'.format(k)] = evaluation(args, model, test_loader_dict['test_super_{}_loader'.format(int(k))], device)
                test_loss_dict['test_family_loss_{}'.format(k)], test_loss_dict['test_family_acc_{}'.format(k)] = evaluation(args, model, test_loader_dict['test_family_{}_loader'.format(int(k))], device)
                # import pdb; pdb.set_trace()
                if test_loss_dict['best_test_fold_acc_{}'.format(k)] < test_loss_dict['test_fold_acc_{}'.format(k)]:
                    test_loss_dict['best_test_fold_acc_{}'.format(k)] = test_loss_dict['test_fold_acc_{}'.format(k)]
                if test_loss_dict['best_test_super_acc_{}'.format(k)] < test_loss_dict['test_super_acc_{}'.format(k)]:
                    test_loss_dict['best_test_super_acc_{}'.format(k)] = test_loss_dict['test_super_acc_{}'.format(k)]
                if test_loss_dict['best_test_family_acc_{}'.format(k)] < test_loss_dict['test_family_acc_{}'.format(k)]:
                    test_loss_dict['best_test_family_acc_{}'.format(k)] = test_loss_dict['test_family_acc_{}'.format(k)]
            t_end_test = time.perf_counter()

            if not save_dir == "" and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if not save_dir == "" and val_acc > best_val_acc:
                print('Saving best val checkpoint ...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
                torch.save(checkpoint, save_dir + '/best_val.pt')
                best_val_acc = val_acc    
                for k in test_lst:
                    test_loss_dict['test_fold_at_best_val_acc_{}'.format(k)] = test_loss_dict['test_fold_acc_{}'.format(k)]
                    test_loss_dict['test_super_at_best_val_acc_{}'.format(k)] = test_loss_dict['test_super_acc_{}'.format(k)]
                    test_loss_dict['test_family_at_best_val_acc_{}'.format(k)] = test_loss_dict['test_family_acc_{}'.format(k)]       

            t_end = time.perf_counter()
            print('Train: Loss:{:.6f} Acc:{:.4f}, Validation: Loss:{:.6f} Acc:{:.4f}, time:{}, train_time:{}, test_time:{}'.format(
                train_loss, train_acc, val_loss, val_acc, 
                t_end-t_start, t_end_train-t_start, t_end_test-t_start_test))
            
            wandb.log({
                'Epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'time': t_end-t_start,
                'best_val_acc': best_val_acc
                }, commit=True)
            for k in test_lst:
                wandb.log({
                    'test_fold_loss_{}'.format(k): test_loss_dict['test_fold_loss_{}'.format(k)],
                    'test_fold_acc_{}'.format(k): test_loss_dict['test_fold_acc_{}'.format(k)],
                    'best_test_fold_acc_{}'.format(k): test_loss_dict['best_test_fold_acc_{}'.format(k)],
                    'test_super_loss_{}'.format(k): test_loss_dict['test_super_loss_{}'.format(k)],
                    'test_super_acc_{}'.format(k): test_loss_dict['test_super_acc_{}'.format(k)],
                    'best_test_super_acc_{}'.format(k): test_loss_dict['best_test_super_acc_{}'.format(k)],
                    'test_family_loss_{}'.format(k): test_loss_dict['test_family_loss_{}'.format(k)],
                    'test_family_acc_{}'.format(k): test_loss_dict['test_family_acc_{}'.format(k)],
                    'best_test_family_acc_{}'.format(k): test_loss_dict['best_test_family_acc_{}'.format(k)],
                    'test_fold_at_best_val_acc_{}'.format(k): test_loss_dict['test_fold_at_best_val_acc_{}'.format(k)],
                    'test_super_at_best_val_acc_{}'.format(k): test_loss_dict['test_super_at_best_val_acc_{}'.format(k)],
                    'test_family_at_best_val_acc_{}'.format(k): test_loss_dict['test_family_at_best_val_acc_{}'.format(k)]}
                    , commit=True)

            scheduler.step()   
            
        writer.close()
        # Save last model
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
        torch.save(checkpoint, save_dir + "/epoch{}.pt".format(epoch))
    
    elif args.dataset == 'PPI':
        print('not supported dataset')

if __name__ == "__main__":
    main()