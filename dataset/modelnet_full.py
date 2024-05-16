from abc import ABCMeta
import ast

import torch
import numpy as np

import matplotlib.pyplot as plt

import sys

import os
from tqdm import tqdm
from torch_geometric.data import Data

from torch_geometric.seed import seed_everything

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import trimesh

import plotly.graph_objects as go


def plot_graph(data, edges, title='untilted_figure'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='grey', alpha=0.01)
    for edge in edges:
        ax.plot([data[edge[0]][0], data[edge[1]][0]], [data[edge[0]][1], data[edge[1]][1]], [data[edge[0]][2], data[edge[1]][2]], color='blue')
    ax.view_init(elev=30, azim=180)
    plt.savefig(f'./out/{title}.png')

# Function to load .off file
def load_off(file):
    with open(file, 'r') as f:
        if 'OFF' != f.readline().strip():
            raise('Not a valid OFF header')
        n_verts, n_faces, _ = tuple([int(s) for s in f.readline().strip().split(' ')])
        verts = [[float(s) for s in f.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in f.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        return np.array(verts), np.array(faces)

def get_edges_from_off(file_path):
    mesh = trimesh.load(file_path)
    edges = mesh.edges_unique
    return edges

def off_to_graph(file_path):
    mesh = trimesh.load(file_path)
    verts = mesh.vertices
    edges = mesh.edges_unique
    return verts, edges

from abc import ABCMeta
import ast

import torch
import numpy as np

sys.path.append('/root/workspace/alignment/pyorbit/utils')


from alignment3D import *
from geometry import *
from data_util import *
from hopcroft import PartitionRefinement
from qhull import Qhull

class Timer(metaclass=ABCMeta):
    def __init__(self, verbose = False, **kwargs):
        self.verbose = verbose
        pass
    def start(self):
        self.st = time.time()
    def stop(self, string=None):
        if self.verbose:
            print(f'{string}{time.time()-self.st}')
        pass


class Frame(metaclass=ABCMeta):
    def __init__(self, tol=1e-2, fast=True, verbose=False, *args, **kwargs):
        super().__init__()
        self.tol = tol
        self.chull = Qhull()
        self.fast = fast
        self.timer = Timer(verbose=verbose)

    def project_sphere(self, data, *args, **kwargs):
        distances = np.linalg.norm(data, axis=1, keepdims=False)
        temp =  data/np.linalg.norm(data, axis=1, keepdims=True)
        arr, key = np.unique(temp, axis=0, return_inverse=True)
        encoding = {}
        dists_hash = {}
        for val in set(key):
            dists = [custom_round(v,self.tol) for v in distances[key==val]]
            dists = tuple(sorted(dists))
            if dists not in dists_hash:
                dists_hash[dists] = id(dists)

            encoding[val] = dists_hash[dists]
        return dists_hash, encoding, arr

    def fast_track(self, data):
        dist = np.linalg.norm(data, axis=1, keepdims=True)
        max_dist = max(dist)
        indices = np.where(np.isclose(max_dist, dist, rtol=self.tol, atol=self.tol))[0]
        mean_pos = 0
        if len(indices)>1: # weak check
            for pos in data[indices]:
                mean_pos += pos
            mean_pos = mean_pos
            normal = planar_normal(mean_pos, np.array([0., 0., 1.]))
            n = (normal / np.linalg.norm(normal))
            proj_on_n = np.dot(data, n)[:, np.newaxis] * n / np.dot(n, n)
            mirrored_data = data - 2 * proj_on_n
        return np.linalg.norm(mirrored_data - data) < self.tol**1.5 * data.shape[0], mean_pos

    def get_frame(self, data, *args, **kwargs):

        self.timer.start()
        data = self.check_type(data) # Assert Type
        data = self.align_center(data) # Assert Centered
        data = data[np.linalg.norm(data, axis=1) > self.tol]
        self.timer.stop(string='Translation invariance: ')
        print(f'Data size({len(data)})')
        # FULL DATA
        # ---------
        self.timer.start()
        # Projection
        # ~~~~~~~~~~
        dist_hash, r_encoding, shell_data = self.project_sphere(data, *args, **kwargs)
        # Convex Hull
        # ~~~~~~~~~~~
        shell_rank = np.linalg.matrix_rank(shell_data, tol=self.tol)
        shell_n = shell_data.shape[0]
        shell_graph = self.chull.get_chull_graph(shell_data, shell_rank, shell_n)
        plot_graph(shell_data, shell_graph, title='full')
        self.timer.stop(string='Sphere embedding: ')

        # SAMPLED DATA
        # ------------
        self.timer.start()
        # Equivalence Classes  
        # ~~~~~~~~~~~~~~~~~~~
        d_class, fp_class, orth_class= self.equivalence_classes(data) # Get equivalence classes
        self.timer.stop(string='Equivalence classes: ')
        print(f'Inv Classes({len(d_class)}), Orth classes({len(orth_class)})')
        # Sampling 
        # ~~~~~~~~
        self.timer.start()
        #rand_key = np.random.choice(list(d_class.keys()))
        #sampled_data = d_class[rand_key]
        sampled_data = []
        for i,values in enumerate(orth_class):
            rand_value_key = np.random.choice(range(len(values)))
            # match the value in the fp_class values
            rand_key = [key for key, value in fp_class.items() if np.allclose(value, values[rand_value_key])][0]
            sampled_data += d_class[rand_key]
        self.timer.stop(string='Sampling: ')
        print(f'Sample size {len(sampled_data)}')
        # Projection
        # ~~~~~~~~~~
        dist_hash, r_encoding, shell_data = self.project_sphere(sampled_data, *args, **kwargs)
        # Convex Hull
        # ~~~~~~~~~~~
        shell_rank = np.linalg.matrix_rank(shell_data, tol=self.tol)
        shell_n = shell_data.shape[0]
        shell_graph = self.chull.get_chull_graph(shell_data, shell_rank, shell_n)
        plot_graph(shell_data, shell_graph, title='sampled')
        self.timer.stop(string='Sphere embedding: ')

        return np.array(sampled_data)

    
    def align_center(self, pointcloud):
        return pointcloud - np.mean(pointcloud,axis=0)

    def equivalence_classes(self, data, *args, **kwargs):
        d_class = {}
        for i,x in enumerate(data):
            # TODO: Return here
            x_norm = custom_round(np.linalg.norm(x),self.tol)
            if x_norm not in d_class:
                d_class[x_norm] = [x]
            else:
                d_class[x_norm] += [x]
        fp_class = {key : np.mean(d_class[key], axis=0) for key in d_class.keys()}
        orth_class = []
        for v in fp_class.values():
            placed = False
            for group in orth_class:
                if all(np.abs(np.dot(v, u)) < self.tol for u in group):
                    group.append(v)
                    placed = True
                    break
            if not placed:
                orth_class.append([v])
        return d_class, fp_class, orth_class


    def project_sphere(self, data, *args, **kwargs):
        distances = np.linalg.norm(data, axis=1, keepdims=False)
        temp =  data/np.linalg.norm(data, axis=1, keepdims=True)
        arr, key = np.unique(temp, axis=0, return_inverse=True)
        encoding = {}
        dists_hash = {}
        for val in set(key):
            dists = [custom_round(v,self.tol) for v in distances[key==val]]
            dists = tuple(sorted(dists))
            if dists not in dists_hash:
                dists_hash[dists] = id(dists)

            encoding[val] = dists_hash[dists]
        return dists_hash, encoding, arr


    def check_type(self, data, *args, **kwargs):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"Data type not supported {type(data)}")


if __name__=='__main__':
    frame = Frame(tol=0.1, fast=True, verbose=True)

    obj = 'table'
    number = 1


    off_file_path = f"/root/workspace/data/ModelNet40/{obj}/train/{obj}_{number:04d}.off"  # Change this to the path of your .off file

    verts, edges = off_to_graph(off_file_path)
    verts =  verts - np.mean(verts,axis=0)
    vmax = np.max(np.abs(verts))
    verts = verts/vmax

    seed_everything(0)


    highlight_points = frame.get_frame(verts)
    #find the fp_isometry with maximum length
    #iso_index = np.argmax([len(isom) for isom in fp_isometries])
    #orth_isometries = np.array(fp_isometries[iso_index])
    #print(orth_isometries)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color='grey', alpha=0.01)
    ax.scatter(highlight_points[:, 0], highlight_points[:, 1], highlight_points[:, 2], color='blue', alpha=.1)
    #ax.scatter(moment[0], moment[1], moment[2], color='red', alpha=1)
    #ax.scatter(orth_isometries[:,0], orth_isometries[:,1], orth_isometries[:,2], color='green', alpha=.2)
    ax.view_init(elev=30, azim=180)
    plt.savefig(f'./out/{obj}_{number}.png')



    #fig = go.Figure(data=[go.Scatter3d(x=orth_isometries[:,0], y=orth_isometries[:,1], z=orth_isometries[:,2], mode='markers', marker=dict(size=2))])
    #fig.write_html(f'./out/{obj}_{number}_isometries.html')
