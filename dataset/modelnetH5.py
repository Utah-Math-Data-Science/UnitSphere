'''
    ModelNet dataset. Support ModelNet40, XYZ channels. Up to 2048 points.
    Faster IO than ModelNetDataset in the first epoch.
'''

from typing import Optional
import time

import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = BASE_DIR
import data_perturbations as provider

import scipy

import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader, Batch
from torch_geometric.transforms import BaseTransform, RadiusGraph, KNNGraph#, Delaunay

# Download dataset for point cloud classification
DATA_DIR = os.path.join(ROOT_DIR, '/root/workspace/data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)


class Delaunay(BaseTransform):
    r"""Computes the delaunay triangulation of a set of points
    (functional name: :obj:`delaunay`).
    """
    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        if data.pos.size(0) < 2:
            data.edge_index = torch.tensor([], dtype=torch.long,
                                           device=data.pos.device).view(2, 0)
        if data.pos.size(0) == 2:
            data.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long,
                                           device=data.pos.device)
        elif data.pos.size(0) == 3:
            data.face = torch.tensor([[0], [1], [2]], dtype=torch.long,
                                     device=data.pos.device)
        if data.pos.size(0) > 3:
            pos = data.pos.cpu().numpy()
            tri = scipy.spatial.Delaunay(pos, qhull_options='QJ')
            vor = scipy.spatial.Voronoi(pos, qhull_options='QJ')
            face = torch.from_numpy(tri.simplices)
            edge = torch.from_numpy(vor.ridge_points)

            data.face = face.t().contiguous().to(data.pos.device, torch.long)
            data.edge_index = edge.t().contiguous().to(data.pos.device, torch.long)
            #data.edge_index = torch.tensor(edge_list, dtype=torch.long, device=data.pos.device).t().contiguous()

        return data

class ModelNetH5Dataset(object):
    def __init__(self, list_filename, batch_size = 32, npoints = 1024, shuffle=True):
        self.list_filename = list_filename
        self.batch_size = batch_size
        self.npoints = npoints
        self.shuffle = shuffle
        self.h5_files = getDataFiles(self.list_filename)
        self.reset()

    def reset(self):
        ''' reset order of h5 files '''
        self.file_idxs = np.arange(0, len(self.h5_files))
        if self.shuffle: np.random.shuffle(self.file_idxs)
        self.current_data = None
        self.current_label = None
        self.current_file_idx = 0
        self.batch_idx = 0
   
    def _augment_batch_data(self, batch_data):
        rotated_data = provider.rotate_point_cloud(batch_data)
        rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:,:,0:3] = jittered_data
        return provider.shuffle_points(rotated_data)


    def _get_data_filename(self):
        return self.h5_files[self.file_idxs[self.current_file_idx]]

    def _load_data_file(self, filename):
        self.current_data,self.current_label = load_h5(filename)
        self.current_label = np.squeeze(self.current_label)
        self.batch_idx = 0
        if self.shuffle:
            self.current_data, self.current_label, _ = shuffle_data(self.current_data,self.current_label)
    
    def _has_next_batch_in_file(self):
        return self.batch_idx*self.batch_size < self.current_data.shape[0]

    def num_channel(self):
        return 3

    def has_next_batch(self):
        # TODO: add backend thread to load data
        if (self.current_data is None) or (not self._has_next_batch_in_file()):
            if self.current_file_idx >= len(self.h5_files):
                return False
            self._load_data_file(self._get_data_filename())
            self.batch_idx = 0
            self.current_file_idx += 1
        return self._has_next_batch_in_file()

    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, self.current_data.shape[0])
        bsize = end_idx - start_idx
        batch_label = np.zeros((bsize), dtype=np.int32)
        data_batch = self.current_data[start_idx:end_idx, 0:self.npoints, :].copy()
        label_batch = self.current_label[start_idx:end_idx].copy()
        self.batch_idx += 1
        if augment: data_batch = self._augment_batch_data(data_batch)
        return data_batch, label_batch 


class ModelNetH5Geometric(InMemoryDataset):
    def __init__(self, root, connectivity, split='train', radius=.1, k=6, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        assert connectivity in ['voronoi', 'knn', 'radius', 'unitsphere']
        self.connectivity = connectivity
        #self.conn_dict = {'knn': KNNGraph(k), 'radius': RadiusGraph(r=radius), 'voronoi': Delaunay(), 'unitsphere': self.frame.get_frame}
        self.conn_dict = {'knn': KNNGraph(k), 'radius': RadiusGraph(r=radius), 'voronoi': Delaunay(), 'unitsphere': Delaunay()}

        super(ModelNetH5Geometric, self).__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.split = split
        if split == 'train':
            self.load(self.processed_paths[0])
        elif split == 'test':
            self.load(self.processed_paths[1])
        else:
            raise ValueError('Split not recognized')
        print(self.data)


    @property
    def raw_file_names(self):
        return ['ModelNet40']

    @property
    def processed_file_names(self):
        return [f'modelnet40_train_data_{self.connectivity}.pt', f'modelnet40_test_data_{self.connectivity}.pt']

    def process(self):
        self.train_loader = ModelNetH5Dataset('/root/workspace/data/modelnet40_ply_hdf5_2048/train_files.txt')
        self.test_loader = ModelNetH5Dataset('/root/workspace/data/modelnet40_ply_hdf5_2048/test_files.txt')

        # train data
        data_list = []
        start = time.time()
        while self.train_loader.has_next_batch():
            bdata, blabel = self.train_loader.next_batch()
            for i in range(bdata.shape[0]):
                data = Data(pos=torch.from_numpy(bdata[i, :, 0:3]).to(torch.float32), y=torch.tensor(blabel[i], dtype=torch.long), z=torch.ones(bdata[i, :, 0:3].shape[0], 1), dtype=torch.long)
                data = self.conn_dict[self.connectivity](data)
                data_list.append(data)
        self.save(data_list, self.processed_paths[0])

        # test data
        data_list = []
        while self.test_loader.has_next_batch():
            bdata, blabel = self.test_loader.next_batch()
            for i in range(bdata.shape[0]):
                data = Data(pos=torch.from_numpy(bdata[i, :, 0:3]).to(torch.float32), y=torch.tensor(blabel[i], dtype=torch.long), z=torch.ones(bdata[i, :, 0:3].shape[0], 1), dtype=torch.long)
                data = self.conn_dict[self.connectivity](data)
                data_list.append(data)

        self.save(data_list, self.processed_paths[1])

        print(f'Processing time: {time.time()-start}')
        pass

def modelnet40_dataloaders(
    connectivity : str = 'radius',
    radius : Optional[float] = None,
    k : Optional[int] = None,
    force_reload : bool = False,
    batch_size : int = 128,
):

    assert(connectivity in ['knn', 'radius', 'voronoi', 'unitsphere']), f'Connectivity not recognized: {connectivity}'
    assert((connectivity!='radius') or (radius is not None)),f'Radial connectivity and radius do not match {connectivity,radius}'
    assert((connectivity!='knn') or (k is not None)),f'KNN connectivity and k do not match {connectivity,k}'

    train_dataset = ModelNetH5Geometric(root='/root/workspace/data/modelnet40_ply_hdf5_2048',
                                        split='train',
                                        connectivity=connectivity,
                                        radius=radius,
                                        k=k,
                                        force_reload=force_reload
    )
    test_dataset = ModelNetH5Geometric(root='/root/workspace/data/modelnet40_ply_hdf5_2048',
                                        split='test',
                                        connectivity=connectivity,
                                        radius=radius,
                                        k=k,
                                        force_reload=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def average_density(datalist):
    edge_count, node_count = [], []
    for data in datalist:
        edge_index = data.edge_index
        edge_count.append(edge_index.shape[1])
        node_count.append(data.num_nodes)
    average_nodes = np.mean(node_count)
    average_edges = np.mean(edge_count)
    return average_edges, average_nodes, edge_count, node_count

if __name__=='__main__':
    #d = ModelNetH5Dataset('/root/workspace/data/modelnet40_ply_hdf5_2048/train_files.txt')
    #print(d.shuffle)
    #print(d.has_next_batch())
    #ps_batch, cls_batch = d.next_batch(True)
    #print(ps_batch.shape)
    #print(cls_batch.shape)
    second_loop = {
            'radius': [{'radius':0.1}, {'radius':0.25}, {'radius':0.5}],
            #'radius': [{'radius':0.1}],
            'knn': [{'k':4}, {'k':16}, {'k':32}],
            'voronoi': [{}],
            'unitsphere': [{}]}

    for connectivity in ['radius', 'knn', 'voronoi', 'unitsphere']:
    #for connectivity in ['radius', 'voronoi']:
        for second in second_loop[connectivity]:
            print('*'*10)
            print(f'Connectivity: {connectivity} ({second})')
            dataset, train_datalist, test_datalist, train_loader, test_loader = modelnet40_dataloaders(connectivity=connectivity, batch_size=32, force_reload=True, **second)
            train_average_edges, train_average_nodes, edge_count, node_count = average_density(train_datalist)
            test_average_edges, test_average_nodes, edge_count, node_count = average_density(test_datalist)
            print('Average density:', f'train ({train_average_edges, train_average_nodes})', f'test ({test_average_edges, test_average_nodes})')
