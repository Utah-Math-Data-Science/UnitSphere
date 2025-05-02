import os
import os.path as osp
import numpy as np
import tqdm
import torch
from sklearn.utils import shuffle
import torch.nn.functional as F
from rdkit import Chem
from mol_unit_sphere import Frame
from torch_geometric.data import Data, DataLoader, InMemoryDataset, download_url, extract_zip

import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
#import plotly.graph_objects as go

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

# conversion = torch.tensor([
#     1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
#     1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
# ])

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV,
    1., 1, 1, 1, 1, 1., 1., 1.
])





class Protein_tmp():
    def __init__(self):
        self.inputdata = [] 
        self.pos = []
        self.edge_list = []

    def read_and_split_file(self, filename):
        """Reads a text file and splits each line by spaces.

        Args:
            filename: The name of the text file.

        Returns:
            A list of lists, where each inner list contains the words of a line.
        """
        with open(filename, 'r') as file:
            lines = file.readlines()

        for line in lines:
            tmp = line.strip().split()
            self.pos.append(tuple(float(tmp[i]) for i in range(1,4)))
        self.pos = np.array(self.pos) 

    def process(self):

        frame = Frame()  

        pos = torch.tensor(self.pos, dtype=torch.float)
        posc = pos - pos.mean(dim=0)

        atomic_number = [0] * len(pos)
        z = torch.tensor(atomic_number, dtype=torch.long)


        pos, z, edge_index_hull, edge_attr_hull, radial_arr = frame.get_frame(pos.numpy(), 
                                                                                z.numpy())
        


        pos = torch.tensor(pos, dtype=torch.float)
        z = torch.tensor(z, dtype=torch.long)
        edge_index_hull = torch.tensor(edge_index_hull, dtype=torch.long)
        edge_attr_hull = torch.tensor(edge_attr_hull, dtype=torch.float)
        
        # if torch.isnan(edge_attr_hull.sum()):
        #     print('Molecule No. {}'.format(i))
        #     print(edge_attr_hull.sum())
        #     break 
        radial_arr = torch.tensor(radial_arr, dtype=torch.float)          
        
        
        self.edge_list = torch.transpose(edge_index_hull, 0, 1).numpy()


    def save_edge_list(self, filename):
        """
        Saves the edge list to a text file.

        Args:
            filename (str): The filename to save the edge list to.
        """
        with open(filename, 'w') as f:
            for edge in self.edge_list:
                # Write each edge element separated by space
                f.write(f"{edge[0]} {edge[1]}\n")

    def load_edge_list(self, filename):
        with open(filename, 'r') as f:
            edges = []
            for line in f:
                # Read each line, split into integers, and append to edges list
                edge = list(map(int, line.strip().split()))
                edges.append(edge)
        self.edge_list = np.array(edges, dtype=int)


    def plot_3d_graph(self, filename=None):
        plt.rcParams.update({'font.size': 8})  # Set font size globally

    # Create a NetworkX graph
        G = nx.Graph()
        for edge in self.edge_list:
            G.add_edge(edge[0], edge[1])

    # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

    # Plot nodes (assuming data.pos is available)
        ax = axes[0]
        ax.scatter(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2])
        ax.set_title("Node View")

        ax = axes[1]
        ax.scatter(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2])
        ax.set_title("Node View")

    # Plot edges with color based on distance
        edge_colors = []
        for u, v in self.edge_list:
            p1, p2 = self.pos[u], self.pos[v]
            distance = np.sqrt(np.sum(np.square(p1 - p2), axis=None))
            edge_colors.append(distance)

        norm = plt.Normalize(min(edge_colors), max(edge_colors))
        edge_colors = plt.cm.coolwarm(norm(edge_colors))

        for i in range(len(self.edge_list)):
            edge = self.edge_list[i]
            x_values = [self.pos[edge[0]][0], self.pos[edge[1]][0]]
            y_values = [self.pos[edge[0]][1], self.pos[edge[1]][1]]
            z_values = [self.pos[edge[0]][2], self.pos[edge[1]][2]]
            ax.plot(x_values, y_values, z_values, color=edge_colors[i])

    # Colorbar
        cax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
        cb = ColorbarBase(cax, cmap=plt.cm.coolwarm, norm=norm, orientation='vertical')
        cb.set_label('Edge Distance')

        if filename:
            plt.savefig(filename)  # Save plot if filename provided
        else:
            plt.show()  # Display plot otherwise

if __name__ == '__main__':

    directory_path = "coarsepdb_UP000005640_9606_HUMAN_v3"
    directory_path_edges = directory_path + "_edge"

    ''' read all txt in directory_path and record them in list_filename '''
    list_filename = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            list_filename.append(filename)
    
    ''' for any filename in list_filename, read the coordinates in the file and then write down the list of edges '''
    with tqdm.tqdm(total=len(list_filename), desc="Processing files") as pbar:
        for filename in list_filename:
            data = Protein_tmp()
            data.read_and_split_file(os.path.join(directory_path, filename))
            data.process()
            data.save_edge_list(os.path.join(directory_path_edges, filename[:-4] + '_edges.txt'))
            #data.plot_3d_graph(filename = filename+'.png')

            pbar.update(1)





