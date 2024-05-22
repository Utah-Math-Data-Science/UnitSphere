#!/usr/bin/python3
"""
ModelNet40
=========

+ Define:     Model
+ Initialize: Config/Model/Dataset
+ Learning:   Train/Validate/Test
+ Drivers:    Main/Hydra/Fold/Train

"""
import os
import time

import hydra
from omegaconf import OmegaConf
import wandb

import torch
from torch.nn import Linear, Module, ReLU, Sequential, SiLU
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import QM9
from torch_geometric.graphgym import global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T

#TODO: Adjust these	imports
import sys
sys.path.append('./dataset/')
from modelnetH5 import modelnet40_dataloaders
sys.path.append('./models/')
from comenet import ComENet
from comenetCHA import ComENetCHA
from schnet import SchNet
from leftnetCHA import LEFTNetCHA

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Model
#----------------------------------------------------------------------------------------------------------------------------------------------------


class Model(Module):
    def __init__(self,
        cutoff=5.0,
        num_layers=4,
        hidden_channels=256,
        middle_channels=64,
        out_channels=1,
        num_radial=3,
        num_spherical=2,
        num_output_layers=3,
        iscovhull = False
    ) -> None:

        super(Model, self).__init__()
        #self.nn = ComENet(iscovhull=False, out_channels=40)
        self.nn = ComENetCHA(out_channels=40)
        #self.nn = LEFTNetCHA(out_channels=40, cutoff=0.05, num_layers=1, hidden_channels=128)


    def forward(self, batch_data):
        out = self.nn(batch_data)
        #out = global_add_pool(out, batch_data.batch)
        out = F.softmax(out, dim=1)
        return out

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Config/Model/Dataset
#----------------------------------------------------------------------------------------------------------------------------------------------------

def setup(cfg):
    # Set device
    args = cfg.setup
    cfg['setup']['device'] = args['device'] if torch.cuda.is_available() else 'cpu'
    os.environ["WANDB_DIR"] = os.path.abspath(args['wandb_dir'])
    # Change file name for sweeping *Prior to setting seed*
    if args['sweep']:
        run_id = wandb.run.id
        cfg['load']['checkpoint_path']=cfg['load']['checkpoint_path'][:-3]+str(run_id)+'.pt'
    # Set Backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pass

#----------------------------------------------------------------------------------------------------------------------------------------------------

def load(cfg):
    args = cfg.load

    train_loader, test_loader = modelnet40_dataloaders(
        connectivity = args['connectivity'],
        radius = args['radius'],
        k = args['k'],
        batch_size = args['batch_size'],
        force_reload = args['force_reload'],
    )

    model_kwargs = OmegaConf.to_container(cfg.model)
    model = Model(
        hidden_channels = model_kwargs['hidden_channels'],
    )

    if os.path.exists(args['checkpoint_path']) and args['load_checkpoint']:
        checkpoint = torch.load(cfg.load['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, train_loader, test_loader

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Train/Validate/Test
#----------------------------------------------------------------------------------------------------------------------------------------------------

def train(cfg, data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(cfg, data, model):
    model.eval()
    output = model(data)
    loss = F.cross_entropy(output.squeeze(), data.y) 
    return loss.item()

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Main/Hydra/Fold/Train
#----------------------------------------------------------------------------------------------------------------------------------------------------

def run_training(cfg, model, train_dl):
    args = cfg.train

    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args['epochs'])

    model = model.to(cfg.setup['device'])


    for epoch in range(args['epochs']):

        model.train()
        train_loss, count = 0, 0
        start = time.time()

        for i,data in enumerate(train_dl):
            data = data.to(cfg.setup['device'])
            batch_loss = train(cfg, data, model, optimizer)

            batch_size = data.y.shape[0]
            train_loss += batch_loss * batch_size
            count += batch_size

            if i%10 == 0:
                print(f'Train({epoch}) | batch({i:03d}) | loss({batch_loss:.4f})')

        end = time.time()
        train_loss = train_loss/count
        scheduler.step()
        
        wandb.log({'epoch':epoch,
            'train_loss':train_loss,
            'time':end-start})
        print(f'Epoch({epoch}) '
            f'| train({train_loss:.4f}) '
            f'| time({end-start:.4f})'
            f'\n')


    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="/root/workspace/UnitSphere/config/", config_name="modelnet40")
def run_modelnet40(cfg):
    """
    Execute run saving details to wandb server.
    """
    wandb.init(entity='utah-math-data-science',
                project='umds-baselines',
                mode='disabled',
                name=cfg.model['name'],
                dir='/root/workspace/out/',
                tags=['modelnet40', cfg.model['name']],
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    
    # Execute
    setup(cfg)
    print(OmegaConf.to_yaml(cfg))
    model, train_dl, test_dl = load(cfg)
    print(model)

    if cfg.setup['train']:
        run_training(cfg, model, train_dl)

    checkpoint = torch.load(cfg.load['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(cfg.setup['device'])

    test_loss, count = 0, 0
    for data in test_dl:
        data.to(cfg.setup['device'])
        batch_loss = test(cfg, data, model)

        batch_size = data.y.shape[0]
        test_loss += batch_loss * batch_size
        count += batch_size
    test_loss = test_loss/count

    print(f'\ntest({test_loss})')
    wandb.log({'test_loss':test_loss})
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_modelnet40()
