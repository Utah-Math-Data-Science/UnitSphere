import urllib.request
import pandas as pd
from typing import Dict, Any
import networkx as nx
from Bio.Data import IUPACData
import graphein.protein as gp
from graphein.protein.edges.distance import add_distance_threshold
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, residue_max_acc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import logging
from alignment.torch_canon.pointcloud import CanonEn as Canon
import numpy as np

DSSP_COLS = [
    "chain",
    "resnum",
    "icode",
    "dssp_index",
    "aa",
    "ss",
    "asa",
    "phi",
    "psi",
    "NH_O_1_relidx",
    "NH_O_1_energy",
    "O_NH_1_relidx",
    "O_NH_1_energy",
    "NH_O_2_relidx",
    "NH_O_2_energy",
    "O_NH_2_relidx",
    "O_NH_2_energy",
    'CA',
    'C',
    'N'
]

DSSP_SS = ["H", "B", "E", "G", "I", "T", "S"]

# Example mapping dictionary for one-letter amino acid codes
one_letter_to_number = {
    "A": 1,  # Alanine
    "R": 2,  # Arginine
    "N": 3,  # Asparagine
    "D": 4,  # Aspartic acid
    "C": 5,  # Cysteine
    "E": 6,  # Glutamic acid
    "Q": 7,  # Glutamine
    "G": 8,  # Glycine
    "H": 9,  # Histidine
    "I": 10, # Isoleucine
    "L": 11, # Leucine
    "K": 12, # Lysine
    "M": 13, # Methionine
    "F": 14, # Phenylalanine
    "P": 15, # Proline
    "S": 16, # Serine
    "T": 17, # Threonine
    "W": 18, # Tryptophan
    "Y": 19, # Tyrosine
    "V": 20, # Valine
}

def map_dssp_num(key: str) -> int:
    if key == '-':
        return 0
    elif key == 'H':
        return 1
    elif key == 'B':
        return 2
    elif key == 'E':
        return 3
    elif key == 'G':
        return 4
    elif key == 'I':
        return 5
    elif key == 'T':
        return 6
    elif key == 'S':
        return 7
    else:
        return 0

def parse_dssp_df(dssp: Dict[str, Any],
                  coords: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse DSSP output to DataFrame

    :param dssp: Dictionary containing DSSP output
    :type dssp: Dict[str, Any]
    :return: pd.Dataframe containing parsed DSSP output
    :rtype: pd.DataFrame
    """
    appender = []
    for k in dssp.keys():
        
        to_append = []
        y = dssp[k]
        chain = k[0]
        residue = k[1]
        # het = residue[0]
        resnum = residue[1]
        icode = residue[2]
        to_append.extend([chain, resnum, icode])
        to_append.extend(y)
        try:
            ca = coords[k]['CA']
            c = coords[k]['C']
            n = coords[k]['N']
            to_append.extend([ca, c, n])
            appender.append(to_append)
        except:
            continue

    return pd.DataFrame.from_records(appender, columns=DSSP_COLS)

# the following funcitons from Graphein has been modified
def process_dssp_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DSSP DataFrame to make indexes align with node IDs

    :param df: pd.DataFrame containing the parsed output from DSSP.
    :type df: pd.DataFrame
    :return: pd.DataFrame with node IDs
    :rtype: pd.DataFrame
    """

    # Convert 1 letter aa code to 3 letter
    amino_acids = df["aa"].tolist()

    for i, amino_acid in enumerate(amino_acids):
        amino_acids[i] = IUPACData.protein_letters_1to3[amino_acid].upper()
    df["aaa"] = amino_acids

    # Construct node IDs
    node_ids = []

    for i, row in df.iterrows():
        node_id = row["chain"] + ":" + row["aaa"] + ":" + str(row["resnum"])
        node_ids.append(node_id)
    df["node_id"] = node_ids

    df.set_index("node_id", inplace=True)

    return df

def add_dssp_feature(G: nx.Graph, feature: str) -> nx.Graph:
    """
    Adds add_dssp_feature specified amino acid feature as calculated
    by DSSP to every node in a protein graph
    :param G: Protein structure graph to add dssp feature to
    :param feature: string specifying name of DSSP feature to add:
    "chain",
    "resnum",
    "icode",
    "dssp_index",
    "aa",   # one letter name of amino acid
    "aaa",  # three letter name of amino acid
    "ss",
    "asa",
    "phi",
    "psi",
    "NH_O_1_relidx",
    "NH_O_1_energy",
    "O_NH_1_relidx",
    "O_NH_1_energy",
    "NH_O_2_relidx",
    "NH_O_2_energy",
    "O_NH_2_relidx",
    "O_NH_2_energy",

    These names parse_dssp_df accessible in the DSSP_COLS list
    :param G: Protein Graph to add features to
    :type G: nx.Graph
    :return: Protein structure graph with DSSP feature added to nodes
    :rtype: nx.Graph
    """

    config = G.graph["config"]
    dssp_df = G.graph["dssp_df"]

    # Change to not allow for atom granularity?
    if config.granularity == "atom":
        # TODO confirm below is not needed and remove
        """
        # If granularity is atom, apply residue feature to every atom
        for n in G.nodes():
            residue = n.split(":")
            residue = residue[0] + ":" + residue[1] + ":" + residue[2]

            G.nodes[n][feature] = dssp_df.loc[residue, feature]
        """
        raise NameError(
            f"DSSP residue features ({feature}) \
            cannot be added to atom granularity graph"
        )

    else:
        nx.set_node_attributes(G, dict(dssp_df[feature]), feature)

    if config.verbose:
        print("Added " + feature + " features to graph nodes")

    return G

# the following function from graphein # has been modified
def number_groups_of_runs(list_of_values: List[Any]) -> List[str]:
    """Numbers groups of runs in a list of values.

    E.g. ``["A", "A", "B", "A", "A", "A", "B", "B"] ->
    ["A1", "A1", "B1", "A2", "A2", "A2", "B2", "B2"]``

    :param list_of_values: List of values to number.
    :type list_of_values: List[Any]
    :return: List of numbered values.
    :rtype: List[str]
    """
    
    df = pd.DataFrame({"val": list_of_values})
    df["idx"] = df["val"].shift() != df["val"]
    df["sum"] = df.groupby("val")["idx"].cumsum()
    return list(df["val"].astype(str) + df["sum"].astype(str))


# Define a function to extract coordinates
def extract_atom_coordinates_as_dict(structure, atoms_of_interest=['CA', 'C', 'N']):
    atom_coordinates = {}
    count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  # Ensures it's a standard amino acid
                    atom_data = {}
                    for atom in residue:
                        if atom.get_name() in atoms_of_interest:
                            atom_data[atom.get_name()] = atom.get_coord()
                    if atom_data:  # Only include if requested atoms are found
                        # Create key as tuple (chain ID, residue ID)
                        key = (chain.id, residue.id)
                        atom_coordinates[key] = atom_data

    return atom_coordinates

def get_pdb_info(pdb_id, chain_id=None):
    
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    file_name = f'{pdb_id}.pdb'
    urllib.request.urlretrieve(url, file_name)

    new_funcs = {
            "edge_construction_functions": [gp.add_peptide_bonds,
                                              gp.add_hydrogen_bond_interactions,
                                              gp.add_disulfide_interactions,
                                              gp.add_ionic_interactions,
                                              gp.add_aromatic_interactions,
                                              gp.add_aromatic_sulphur_interactions,
                                              gp.add_cation_pi_interactions],
             "dssp_config": gp.DSSPConfig()
            }
    config = gp.ProteinGraphConfig(**new_funcs)
    g = gp.construct_graph(config=config, pdb_code= pdb_id)

    # Parse the PDB file and run DSSP
    parser = PDBParser()
    structure = parser.get_structure('protein', file_name)
    model = structure[0]
    dssp = DSSP(model, file_name)
    dssp_dict = {}
    
    # Extract coordinates of C_alpha, C, and N atoms
    atoms_of_interest = ['CA', 'C', 'N']  # List of atom names to extract
    atom_coordinates = extract_atom_coordinates_as_dict(structure, atoms_of_interest)

    for key in dssp.keys():
        dssp_dict[key] = dssp[key]

    dssp_dict = parse_dssp_df(dssp_dict, atom_coordinates)
    dssp_dict = process_dssp_df(dssp_dict)
    
    if chain_id:
        df = dssp_dict[dssp_dict['chain'] == chain_id]
    else:
        df = dssp_dict
    df = df.reset_index()
    
    edge_dict = {}
    for i in range(len(df)):
        edge_dict[i] = []
        
    node_lst = list(df['node_id'])
    for k in g.edges:
        k_ = {}
        k_[0] = k[0].split(':')[0] + ':' + k[0].split(':')[1] + ':' + k[0].split(':')[2]
        k_[1] = k[1].split(':')[0] + ':' + k[1].split(':')[1] + ':' + k[1].split(':')[2]
        if chain_id == k[0].split(':')[0] and chain_id == k[1].split(':')[0]:
            if k[0] in node_lst and k[1] in node_lst:
                id_0 = df.loc[df['node_id'] == k_[0]].index[0]
                id_1 = df.loc[df['node_id'] == k_[1]].index[0]
                edge_dict[id_0].append(id_1)
                edge_dict[id_1].append(id_0)

    # sort each list 
    edge_lst0 = []
    edge_lst1 = []
    for i in range(len(df)):
        edge_dict[i] = sorted(edge_dict[i])
        if len(edge_dict[i]) > 0:
            for j in edge_dict[i]:
                edge_lst0.append(i)
                edge_lst1.append(j)
    ss_ser = []
    ss_count = 0
    for i in range(len(df)):
        if i == 0:
            ss_ser.append(ss_count)
        else:
            if df['ss'].iloc[i] == df['ss'][i-1]:
                ss_ser.append(ss_count)
            else:
                ss_count += 1
                ss_ser.append(ss_count)
    df['ss_ser'] = pd.Series(ss_ser)
    df['ss_num'] = df['ss'].apply(map_dssp_num)

    ss_coords = {}
    l_ss_ser = max(df['ss_ser'])
    s_graphs_dict = {}
    s_graphs_dict['ss_ser'] = []
    s_graphs_dict['ss_num'] = []
    s_graphs_dict['coords'] = []
    for i in range(l_ss_ser+1):
        df_sub = df[df['ss_ser'] == i]
        vectors = np.array(df_sub['CA'].tolist())
        coords = np.mean(vectors, axis=0).tolist()
        s_graphs_dict['coords'].append(coords)
        s_graphs_dict['ss_ser'].append(df_sub['ss_ser'].iloc[0])
        s_graphs_dict['ss_num'].append(df_sub['ss_num'].iloc[0])
        

    return df, [edge_lst0, edge_lst1], s_graphs_dict

if __name__ == '__main__':
    pdb_id = '1nxk'
    chain_id = 'B'
    get_pdb_info(pdb_id, chain_id)