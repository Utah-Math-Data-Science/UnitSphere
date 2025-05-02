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
from schnet import SchNet

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Model
#----------------------------------------------------------------------------------------------------------------------------------------------------


class Model(Module):
    def __init__(self,
        input_channels,
        edge_attr_dim,
        hidden_channels,
        act_fn=SiLU(),
        n_layers=4,
        coords_weight=1.0,
        attention=False,
        node_attr=1
    ) -> None:

        super(Model, self).__init__()
        self.hidden_nf = hidden_channels
        self.n_layers = n_layers
        self.node_attr = node_attr

        # Encoder
        self.embedding = Linear(input_channels, hidden_channels)

        # Message Passing Layers
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, SchNet(self.hidden_nf, self.hidden_nf))
        
        # Decoders
        self.node_dec = Sequential(Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = Sequential(Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       Linear(self.hidden_nf, 1))

    def forward(self, h0, x, edges, edge_attr, batch):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_attr=h0)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_attr=None)

        h = self.node_dec(h)
        h = global_add_pool(h,batch)
        pred = self.graph_dec(h)
        return pred.squeeze(1)

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Helper
#----------------------------------------------------------------------------------------------------------------------------------------------------

def compute_mean_mad(data):
    values = data.y
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)
    return meann, mad

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

    dataset = modelnet40_dataloaders(
        connectivity = args['connectivity'],
        radius = args['radius'],
        k = args['k'],
        batch_size = args['batch_size'],
    )

    model_kwargs = OmegaConf.to_container(cfg.model)
    model = Model(
        input_channels = dataset.num_features,
        edge_attr_dim = 0,
        hidden_channels = model_kwargs['hidden_channels'],
        n_layers = model_kwargs['hidden_layers'],
        coords_weight = 1.0,
        attention = model_kwargs['attention'],
        node_attr = model_kwargs['node_attr']
    )

    if os.path.exists(args['checkpoint_path']) and args['load_checkpoint']:
        checkpoint = torch.load(cfg.load['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, train_dl, val_dl, test_dl

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Train/Validate/Test
#----------------------------------------------------------------------------------------------------------------------------------------------------

def train(cfg, data, model, optimizer, meann, mad):
    # meann, mad = compute_mean_mad(data)
    model.train()
    optimizer.zero_grad()
    output = model(h0=data.x, x=data.pos, edges=data.edge_index, edge_attr=None, batch=data.batch)
    loss = F.l1_loss(output, (data.y - meann)/mad)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def validate(cfg, data, model, meann, mad):
    # meann, mad = compute_mean_mad(data)
    model.eval()
    output = model(h0=data.x, x=data.pos, edges=data.edge_index, edge_attr=None, batch=data.batch)
    loss = F.l1_loss(mad * output + meann, data.y) 
    return loss.item()

@torch.no_grad()
def test(cfg, data, model, meann, mad):
    # meann, mad = compute_mean_mad(data)
    model.eval()
    output = model(h0=data.x, x=data.pos, edges=data.edge_index, edge_attr=None, batch=data.batch)
    loss = F.l1_loss(mad * output + meann, data.y) 
    return loss.item()

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Main/Hydra/Fold/Train
#----------------------------------------------------------------------------------------------------------------------------------------------------

def run_training(cfg, model, train_dl, val_dl):
    args = cfg.train

    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args['epochs'])

    model = model.to(cfg.setup['device'])

    meann, mad = train_dl.dataset.data['meann'], train_dl.dataset.data['mad']

    best = 1e8
    for epoch in range(args['epochs']):

        model.train()
        train_loss, count = 0, 0
        start = time.time()

        for i,data in enumerate(train_dl):
            data = data.to(cfg.setup['device'])
            batch_loss = train(cfg, data, model, optimizer, meann, mad)

            batch_size = data.y.shape[0]
            train_loss += batch_loss * batch_size
            count += batch_size

            if i%10 == 0:
                print(f'Train({epoch}) | batch({i:03d}) | loss({batch_loss:.4f})')

        end = time.time()
        train_loss = train_loss/count
        scheduler.step()
        
        model.eval()
        val_loss, count = 0, 0
        for i,data in enumerate(val_dl): 
            data = data.to(cfg.setup['device'])
            batch_loss = validate(cfg, data, model, meann, mad)

            batch_size = data.y.shape[0]
            val_loss += batch_loss * batch_size
            count += batch_size

            if i%10 == 0:
                print(f'Valid({epoch}) | batch({i:03d}) | loss({batch_loss:.4f})')

        val_loss = val_loss/count
        perf_metric = val_loss #your performance metric here
        lr = optimizer.param_groups[0]['lr']

        if perf_metric < best:
            best = perf_metric
            bad_itr = 0
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': lr,
                'loss': val_loss,
                },
                cfg.load['checkpoint_path']
            )
        else:
            bad_itr += 1

        wandb.log({'epoch':epoch,
            'train_loss':train_loss,
            'val_loss':val_loss,
            'best':best,
            'lr':lr,
            'time':end-start})
        print(f'Epoch({epoch}) '
            f'| train({train_loss:.4f}) '
            f'| val({val_loss:.4f}) '
            f'| lr({lr:.2e}) '
            f'| best({best:.4f}) '
            f'| time({end-start:.4f})'
            f'\n')

        if bad_itr>args['patience']:
            break

    return best

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
    model, train_dl, val_dl, test_dl = load(cfg)
    meann, mad = train_dl.dataset.data['meann'], train_dl.dataset.data['mad']
    print(model)

    if cfg.setup['train']:
        run_training(cfg, model, train_dl, val_dl)

    checkpoint = torch.load(cfg.load['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(cfg.setup['device'])

    test_loss, count = 0, 0
    for data in test_dl:
        data.to(cfg.setup['device'])
        batch_loss = test(cfg, data, model, meann, mad)

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
