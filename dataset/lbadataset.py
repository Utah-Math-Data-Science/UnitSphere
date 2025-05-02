import torch, random, scipy, math
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from atom3d.datasets import LMDBDataset
import sys
sys.path.append('/root/workspace/UnitSphere/')
from mol_unit_sphere import Frame
from torch.utils.data import IterableDataset
import torch_cluster, torch_geometric, torch_scatter

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, Dataset
import os.path as osp
import h5py
from tqdm import tqdm
import lmdb
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import os

_NUM_ATOM_TYPES = 9
_element_mapping = lambda x: {
    'H' : 0,
    'C' : 1,
    'N' : 2,
    'O' : 3,
    'F' : 4,
    'S' : 5,
    'Cl': 6, 'CL': 6,
    'P' : 7
}.get(x, 8)
lst_amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 
                   'CYS', 'GLU', 'GLN', 'GLY', 
                   'HIS', 'ILE', 'LEU', 'LYS', 
                   'MET', 'PHE', 'PRO', 'SER', 
                   'THR', 'TRP', 'TYR', 'VAL']
_amino_acids = lambda x: {
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLU': 5,
    'GLN': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19
}.get(x, 20)
num_amino_acids = lambda x: {
    0: 'ALA',
    1: 'ARG',
    2: 'ASN',
    3: 'ASP',
    4: 'CYS',
    5: 'GLU',
    6: 'GLN',
    7: 'GLY',
    8: 'HIS',
    9: 'ILE',
    10: 'LEU',
    11: 'LYS',
    12: 'MET',
    13: 'PHE',
    14: 'PRO',
    15: 'SER',
    16: 'THR',
    17: 'TRP',
    18: 'TYR',
    19: 'VAL'
}.get(x, 'UNK')

_DEFAULT_V_DIM = (100, 16)
_DEFAULT_E_DIM = (32, 1)

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device='cpu'):
    
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), 
               D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num,
            (edge_s, edge_v))

    return edge_s, edge_v

class BaseTransform:
    '''
    Implementation of an ATOM3D Transform which featurizes the atomic
    coordinates in an ATOM3D dataframes into `torch_geometric.data.Data`
    graphs. This class should not be used directly; instead, use the
    task-specific transforms, which all extend BaseTransform. Node
    and edge features are as described in the EGNN manuscript.
    
    Returned graphs have the following attributes:
    -x          atomic coordinates, shape [n_nodes, 3]
    -atoms      numeric encoding of atomic identity, shape [n_nodes]
    -edge_index edge indices, shape [2, n_edges]
    -edge_s     edge scalar features, shape [n_edges, 16]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    
    Subclasses of BaseTransform will produce graphs with additional 
    attributes for the tasks-specific training labels, in addition 
    to the above.
    
    All subclasses of BaseTransform directly inherit the BaseTransform
    constructor.
    
    :param edge_cutoff: distance cutoff to use when drawing edges
    :param num_rbf: number of radial bases to encode the distance on each edge
    :device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, edge_cutoff=4.5, num_rbf=16, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device
            
    def __call__(self, df):
        '''
        :param df: `pandas.DataFrame` of atomic coordinates
                    in the ATOM3D format
        
        :return: `torch_geometric.data.Data` structure graph
        '''
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(),
                                     dtype=torch.float32, device=self.device)
            atoms = torch.as_tensor(list(map(_element_mapping, df.element)),
                                            dtype=torch.long, device=self.device)

            edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff)

            edge_s, edge_v = _edge_features(coords, edge_index, 
                                D_max=self.edge_cutoff, num_rbf=self.num_rbf, device=self.device)

            return torch_geometric.data.Data(x=coords, atoms=atoms,
                        edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)


class LBATransform(BaseTransform):
    '''
    Transforms dict-style entries from the ATOM3D LBA dataset
    to featurized graphs. Returns a `torch_geometric.data.Data`
    graph with attribute `label` for the neglog-affinity
    and all structural attributes as described in BaseTransform.
    
    The transform combines the atomic coordinates of the pocket
    and ligand atoms and treats them as a single structure / graph. 
    
    Includes hydrogen atoms.
    '''
    def __call__(self, elem):
        pocket, ligand = elem['atoms_pocket'], elem['atoms_ligand']
        df = pd.concat([pocket, ligand], ignore_index=True)
        
        data = super().__call__(df)
        with torch.no_grad():
            data.label = elem['scores']['neglog_aff']
            lig_flag = torch.zeros(df.shape[0], device=self.device, dtype=torch.bool)
            lig_flag[-len(ligand):] = 1
            data.lig_flag = lig_flag
        return data
    
class LBADataset(InMemoryDataset):
    def __init__(self, 
                 root, 
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None,
                 edge_cutoff=10, 
                 num_rbf=16, 
                 device='cpu'):
        self.device = device
        self.root = root
        self.split = root.split('/')[-1]   
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.frame = Frame()
        super(LBADataset, self).__init__(
            root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.data_p, self.slices_p = torch.load(self.processed_paths[1])

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, name, self.split)
                                
    @property
    def raw_file_names(self):
        name = self.split + '.txt'
        return name

    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def _normalize(self,tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))
    
    def _get_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        

        mask_n = np.char.equal(atom_names, 'N')
        mask_ca = np.char.equal(atom_names, 'CA')
        mask_c = np.char.equal(atom_names, 'C')
        mask_cb = np.char.equal(atom_names, 'CB')
        mask_g = np.char.find(atom_names, 'G') != -1
        mask_d = np.char.find(atom_names, 'D') != -1
        mask_e = np.char.find(atom_names, 'E') != -1
        mask_z = np.char.find(atom_names, 'Z') != -1
        mask_h = np.char.find(atom_names, 'H') != -1
        # mask_g = np.char.equal(atom_names, 'CG') | np.char.equal(atom_names, 'SG') | np.char.equal(atom_names, 'OG') | np.char.equal(atom_names, 'CG1') | np.char.equal(atom_names, 'OG1')
        # mask_d = np.char.equal(atom_names, 'CD') | np.char.equal(atom_names, 'SD') | np.char.equal(atom_names, 'CD1') | np.char.equal(atom_names, 'OD1') | np.char.equal(atom_names, 'ND1')
        # mask_e = np.char.equal(atom_names, 'CE') | np.char.equal(atom_names, 'NE') | np.char.equal(atom_names, 'OE1')
        # mask_z = np.char.equal(atom_names, 'CZ') | np.char.equal(atom_names, 'NZ')
        # mask_h = np.char.equal(atom_names, 'HN1')
        
        pos_n = np.full((len(amino_types),3), np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types),3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types),3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types),3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types),3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types),3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types),3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h
    
    def _compute_dihedrals(self, v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion
    
    def _side_chain_embs(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self._compute_dihedrals(v1, v2, v3),1)
        angle2 = torch.unsqueeze(self._compute_dihedrals(v2, v3, v4),1)
        angle3 = torch.unsqueeze(self._compute_dihedrals(v3, v4, v5),1)
        angle4 = torch.unsqueeze(self._compute_dihedrals(v4, v5, v6),1)
        angle5 = torch.unsqueeze(self._compute_dihedrals(v5, v6, v7),1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
        
        return side_chain_embs
    
    def _bb_embs(self, X):   
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6 
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self._compute_dihedrals(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features
    
    def _transform(self, lmd_data):
        
        data = Data()
        df = lmd_data['atoms_protein']
        df = df[df['resname'].isin(lst_amino_acids)].reset_index(drop=True)
        atom_amino_id_lst = []
        amino_types_lst = []
        ct = 0
        for i in range(len(df['residue'])):
            if i == 0:
                atom_amino_id_lst.append(ct)
                amino_types_lst.append(_amino_acids(df['resname'][i]))
            elif df['residue'][i] != df['residue'][i-1]:
                ct += 1
                atom_amino_id_lst.append(ct)
                amino_types_lst.append(_amino_acids(df['resname'][i]))
            else:
                atom_amino_id_lst.append(ct)
        df['atom_amino_id'] = atom_amino_id_lst
        drop_ca_lst = []
        drop_atypes_lst = []
        for i in range(ct+1):
            if 'CA' not in np.array(df[df['atom_amino_id']==i]['fullname']):
                drop_ca_lst.append(df[df['atom_amino_id']==i].index)
                drop_atypes_lst.append(i)
        for index in drop_ca_lst:
            df = df.drop(index)
        df = df.reset_index(drop=True)
        k = 0 
        atom_amino_id_lst = []
        for i in range(len(df)):
            if i == 0:
                atom_amino_id_lst.append(k)
            elif df['atom_amino_id'][i] != df['atom_amino_id'][i-1]:
                k += 1
                atom_amino_id_lst.append(k)
            else:
                atom_amino_id_lst.append(k)
        df['atom_amino_id'] = atom_amino_id_lst
        amino_types_lst = [item for i, item in enumerate(amino_types_lst) if i not in drop_atypes_lst]

        amino_types = np.array(amino_types_lst, dtype=int)
        atom_names = np.array(df['fullname'].values, dtype=str)
        atom_amino_id = np.array(df['atom_amino_id'].values, dtype=int)
        atom_pos = np.array(df[['x', 'y', 'z']].values, dtype=float)
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = self._get_atom_pos(amino_types, atom_names, atom_amino_id, atom_pos)
        
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        side_chain_embs = self._side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0
        data.side_chain_embs = side_chain_embs
        
        # three backbone torsion angles
        bb_embs = self._bb_embs(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
        bb_embs[torch.isnan(bb_embs)] = 0
        data.bb_embs = bb_embs

        data.x = torch.unsqueeze(torch.tensor(amino_types),1)
        data.coords_ca = pos_ca
        data.coords_n = pos_n
        data.coords_c = pos_c
        assert len(data.x)==len(data.coords_ca)==len(data.coords_n)==len(data.coords_c)==len(data.side_chain_embs)==len(data.bb_embs)

        # egde features
        edge_index = torch_cluster.radius_graph(pos_ca, r=self.edge_cutoff)
        edge_s, edge_v = _edge_features(pos_ca, edge_index, D_max=self.edge_cutoff, num_rbf=self.num_rbf, device=self.device)
        _, shell_data_ch, edge_index_hull = self.frame.get_frame(pos_ca.numpy())
        edge_index_hull = torch.tensor(edge_index_hull, dtype=torch.long, device=self.device)
        ch_pos = torch.tensor(shell_data_ch, dtype=torch.float)
        ch_r = torch.norm(ch_pos - torch.mean(ch_pos, dim=0), dim=-1)
        ch_edge_s, ch_edge_v = _edge_features(ch_pos, edge_index_hull, D_max=self.edge_cutoff, num_rbf=self.num_rbf, device=self.device)

        # 赋值
        data.edge_index = edge_index
        data.edge_s = edge_s
        data.edge_v = edge_v
        data.ch_edge_index = edge_index_hull
        data.ch_pos = ch_pos
        data.ch_r = ch_r
        data.atoms = side_chain_embs.mean(dim=-1)
        data.ch_edge_s = ch_edge_s
        data.ch_edge_v = ch_edge_v

        pocket, ligand = lmd_data['atoms_pocket'], lmd_data['atoms_ligand']
        df_label = pd.concat([pocket, ligand], ignore_index=True)
        data.label = lmd_data['scores']['neglog_aff']
        lig_flag = torch.zeros(df_label.shape[0], device=self.device, dtype=torch.bool)
        lig_flag[-len(ligand):] = 1
        data.lig_flag = lig_flag
        
        return data
    
    
    def process(self):
        
        parent_path = os.path.dirname(self.root)
        ss = 'test'
        self.root = os.path.join(parent_path, ss)
        
        dataset = LMDBDataset(self.root)
        data_list = []
        n_nodes = []
        err_lst = []
        ct = -1
        
        dict_ = {}
        dict_['pdb_id'] = []
        dict_['label'] = []
        for data in tqdm(dataset):
            try:
                dict_['pdb_id'].append(data['id'])
                dict_['label'].append(data['scores']['neglog_aff'])
            except:
                continue
            # try:
            #     data = self._transform(data)
            # except:
            #     print('Error in processing data: {}'.format(ct))
            #     err_lst.append(ct)
            #     continue
            # data_list.append(data)
            # n_nodes.append(data.coords_ca.shape[0])
        df = pd.DataFrame(dict_)
        df.to_csv('lba_{}_pdb_label.csv'.format(ss))
        import pdb; pdb.set_trace()
        print('丢包率: {:.4f}%'.format(len(err_lst)/len(dataset)*100))
        plt.figure()
        plt.hist(n_nodes, bins=200)
        plt.xlabel('# of Node')
        plt.ylabel('# of Graph')
        save_path = osp.join(self.root, '{}_n_nodes_hist.pdf'.format(self.split))
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Done! ------------------------------------')
