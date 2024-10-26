import itertools
from collections import defaultdict
from operator import neg
import random
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from rdkit import Chem
import pandas as pd
import numpy as np

data_dir = "data"

df_drugs_smiles = pd.read_csv(f'{data_dir}/drug_smiles.csv')

DRUG_TO_INDX_DICT = {drug_id: indx for indx, drug_id in enumerate(df_drugs_smiles['drug_id'])}

# drug_id_mol_graph_tup (list of tuple): Contains drug information of all drugs as a list of tuples.
# each tuple has the following elements:
# - id: drug ID provided in drug_smiles.csv
# - mol_graph: rdkit Mol object extracted from Smiles string
# note: smiles object only represent the connectivity of atoms in a molecule in a text form.
#       where Rdkit Mol object capture a complete graph-based representation of molecule.
#       making it easier for feature extraction (atomic features, bond features, ...)
drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip())) for id, smiles in zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])]

# ATOM_MAX_NUM (int): Max number of atoms in every drugs in drug_id_mol_graph_tup list.
ATOM_MAX_NUM = np.max([m[1].GetNumAtoms() for m in drug_id_mol_graph_tup])

# AVAILABLE_ATOM_SYMBOLS (list of str): Contains all symbols of all atoms of each drugs in drug_id_mol_graph_tup list.
# atoms have symbols like Ag, Na, Br, ...
AVAILABLE_ATOM_SYMBOLS = list({a.GetSymbol() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})

# AVAILABLE_ATOM_DEGREES (list of int): Contains all degree of all atoms of all drugs in drug_id_mol_graph_tup list
# an atom degree is the total number of connections between atoms (Simply it's degree of nodes).
AVAILABLE_ATOM_DEGREES = list({a.GetDegree() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})

# AVAILABLE_ATOM_TOTAL_HS (list of int): Contains all Hydrogens attached to the atoms of all drugs in drug_id_mol_graph_tup list
AVAILABLE_ATOM_TOTAL_HS = list({a.GetTotalNumHs() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})

# max_valence (int): A maximum number of Implicit Valence in all atoms of all drugs. Minimum = 9.
# implicit valance: describes the "wants" of complete bonds. For eg. C=0 wants to complete 4 bonds,
#                   because C can form 4 bonds with other atoms or molecule
#                   Tthen Implicit Valance = 2.
max_valence = max(a.GetImplicitValence() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup))
max_valence = max(max_valence, 9)

# AVAILABLE_ATOM_VALENCE (NumPy array of int): generates a NumPy array of integers, starting from 0 and ending at value. 
# the result is an array containing all integers from 0 up to max_valence (but not including max_valence).
AVAILABLE_ATOM_VALENCE = np.arange(max_valence + 1)

# MAX_ATOM_FC (int): maximum absolute formal charge of all atoms in a list of molecular structures.
# formal Charge: Provide a way to estimate ad track electron distribution within a molecule.
MAX_ATOM_FC = abs(np.max([a.GetFormalCharge() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_ATOM_FC = MAX_ATOM_FC if MAX_ATOM_FC else 0

# MAX_RADICAL_ELC (int): calculates the maximum absolute number of radical electrons found across all atoms in a list of molecular structures.
# radical electrons: this is important feature, play importan participation in reactions. 
#                    this is the key in many chemical.
MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_RADICAL_ELC = MAX_RADICAL_ELC if MAX_RADICAL_ELC else 0


def one_of_k_encoding_unk(x, allowable_set):
    """
    Get all matched elements in allowable set.

    Args:
        x (any): element to check.
        allowable_set (list of any): List of any elements are allowed.

    Returns:
        list of any: Return list of any elements in allowable_set equal to x. 
            If x not in allowable_set then x = last element of the allowable set.
    """
    if x not in allowable_set:
        # This would be Unknown
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom,
                explicit_H=True,
                use_chirality=False,
                use_explicit_valence=True):
    """
    Get all features of an atom.

    Args:
        atom (Atom): atom to be performed features extractation.
        explicit_H (boolean): include Explicit Hydrogens. Default = True.
        use_chirality (boolean): include Chirality feature. Default = False.
        use_explicit_valence (boolean): include Explicit Valence feature. Default = True.

    Returns:
        NumPy array: Represents an one hot encoding feature vector of an atom.
    """

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(), 
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # in case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    # chilarity: chilarity molecule can have very different properties and biological effects depending on
    # their orientation. This influence how the molecule interact with biological systems or other chemical entities.
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    if use_explicit_valence:
        results = results + [atom.GetExplicitValence()]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)

def get_mol_edge_list_and_feat_mtx(mol_graph):
    """
    Get molecule edge list and features matrix

    Args:
        mol_graph (Mol): rdkit mol object of a drug.

    Returns:
        tuple: edge list and features matrix
    """
    # features (list of tuple): Contains drug id and features of all atom as a list of tuples.
    # each tuple has the following elements:
    # - index: index of an atom.
    # - features (Numpy array): features of an atom
    features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]

    # to make sure that the feature matrix is aligned according to the idx of the atom
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)

    # GetBonds(): retrive all bonds in the Mol object. Each bond represent connections between 2 atoms.
    # for each bond, retrieve the indices of 2 connected by a bond to form a tuple (start_atom_index, end_atom_index)
    # representing edge between 2 atoms.
    # convert to a Torch Tensor (2D tensor) of a matrix.
    # each row represent an edge, the first column and second column being the indices of the bonded atoms.
    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    
    return undirected_edge_list.T, features


# MOL_EDGE_LIST_FEAT_MTX (dict): each field in the dict is (drug_id: (edge_list, feature_matrix))
# edge_list (shape: (num_edges, 2)): in the mol object, retrive all bonds. "bonds" means the connection between atoms.
#           list of edges tuple(atom1 index, atom2 index).
# undirected_edge_list (shape: (2, 2 * num_edges)): since molecular bonds are typically undirected, we make each directed edge bidirectional.
#                      first row ontains the start nodes of each edge, second row contains the end nodes of each edge.
# undirected_edge_list.T (shape (2*num_edges, 2)): 2 columns, first col is start node(atom1) and second one is end node(atom2).

MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol) 
                                for drug_id, mol in drug_id_mol_graph_tup}
MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}

# TOTAL_ATOM_FEATS (int): total features of an atom.
TOTAL_ATOM_FEATS = (next(iter(MOL_EDGE_LIST_FEAT_MTX.values()))[1].shape[-1])


##### DDI statistics and counting #######
df_all_pos_ddi = pd.read_csv(f'{data_dir}/ddis.csv')
all_pos_tup = [(h, t, r) for h, t, r in zip(df_all_pos_ddi['d1'], df_all_pos_ddi['d2'], df_all_pos_ddi['type'])]

# ALL_DRUG_IDS (Numpy array of int): list of all drug ids
ALL_DRUG_IDS, _ = zip(*drug_id_mol_graph_tup)
ALL_DRUG_IDS = np.array(list(set(ALL_DRUG_IDS)))

# ALL_TRUE_H_WITH_TR (dict): all drug 1 with drug 2 and their relationship
ALL_TRUE_H_WITH_TR = defaultdict(list)

# ALL_TRUE_T_WITH_HR (dict): all drug 2 with drug 1 and their relationship
ALL_TRUE_T_WITH_HR = defaultdict(list)

# FREQ_REL (dict): calculate number of relationships appeared.
FREQ_REL = defaultdict(int)

# ALL_H_WITH_R (dict): mark the dict relationships with head drug.
ALL_H_WITH_R = defaultdict(dict)

# ALL_T_WITH_R (dict): mark the dict relationships with tail drug.
ALL_T_WITH_R = defaultdict(dict)

# ALL_TAIL_PER_HEAD (dict): freaquent of relationships / (length of list drug tail in relationships "r")
ALL_TAIL_PER_HEAD = {}

# ALL_HEAD_PER_TAIL (dict): freaquent of relationships / (length of list drug head in relationships "r")
ALL_HEAD_PER_TAIL = {}

for h, t, r in all_pos_tup:
    ALL_TRUE_H_WITH_TR[(t, r)].append(h)
    ALL_TRUE_T_WITH_HR[(h, r)].append(t)
    FREQ_REL[r] += 1.0
    ALL_H_WITH_R[r][h] = 1
    ALL_T_WITH_R[r][t] = 1

for t, r in ALL_TRUE_H_WITH_TR:
    ALL_TRUE_H_WITH_TR[(t, r)] = np.array(list(set(ALL_TRUE_H_WITH_TR[(t, r)])))
for h, r in ALL_TRUE_T_WITH_HR:
    ALL_TRUE_T_WITH_HR[(h, r)] = np.array(list(set(ALL_TRUE_T_WITH_HR[(h, r)])))

for r in FREQ_REL:
    ALL_H_WITH_R[r] = np.array(list(ALL_H_WITH_R[r].keys()))
    ALL_T_WITH_R[r] = np.array(list(ALL_T_WITH_R[r].keys()))
    ALL_HEAD_PER_TAIL[r] = FREQ_REL[r] / len(ALL_T_WITH_R[r])
    ALL_TAIL_PER_HEAD[r] = FREQ_REL[r] / len(ALL_H_WITH_R[r])

#######    ****** ###############

class DrugDataset(Dataset):
    def __init__(self, tri_list, ratio=1.0,  neg_ent=1, disjoint_split=True, shuffle=True):
        ''''disjoint_split: Consider whether entities should appear in one and only one split of the dataset
        ''' 
        self.neg_ent = neg_ent
        self.tri_list = []
        self.ratio = ratio

        for h, t, r, *_ in tri_list:
            if ((h in MOL_EDGE_LIST_FEAT_MTX) and (t in MOL_EDGE_LIST_FEAT_MTX)):
                self.tri_list.append((h, t, r))

        if disjoint_split:
            d1, d2, *_ = zip(*self.tri_list)
            self.drug_ids = np.array(list(set(d1 + d2)))
        else:
            self.drug_ids = ALL_DRUG_IDS

        self.drug_ids = np.array([id for id in self.drug_ids if id in MOL_EDGE_LIST_FEAT_MTX])
        
        if shuffle:
            random.shuffle(self.tri_list)
        limit = math.ceil(len(self.tri_list) * ratio)
        self.tri_list = self.tri_list[:limit]

    def __len__(self):
        return len(self.tri_list)
    
    def __getitem__(self, index):
        return self.tri_list[index]

    def collate_fn(self, batch):

        pos_rels = []
        pos_h_samples = []
        pos_t_samples = []
        neg_rels = []
        neg_h_samples = []
        neg_t_samples = []

        for h, t, r in batch:
            pos_rels.append(r)
            h_data = self.__create_graph_data(h)
            t_data = self.__create_graph_data(t)
            pos_h_samples.append(h_data)
            pos_t_samples.append(t_data)

            neg_heads, neg_tails = self.__normal_batch(h, t, r, self.neg_ent)

            for neg_h in neg_heads:
                neg_rels.append(r)
                neg_h_samples.append(self.__create_graph_data(neg_h))
                neg_t_samples.append(t_data)

            for neg_t in neg_tails:
                neg_rels.append(r)
                neg_h_samples.append(h_data)
                neg_t_samples.append(self.__create_graph_data(neg_t))

        pos_h_samples = Batch.from_data_list(pos_h_samples)
        pos_t_samples = Batch.from_data_list(pos_t_samples)
        pos_rels = torch.LongTensor(pos_rels)
        pos_tri = (pos_h_samples, pos_t_samples, pos_rels)

        neg_h_samples = Batch.from_data_list(neg_h_samples)
        neg_t_samples = Batch.from_data_list(neg_t_samples)
        neg_rels = torch.LongTensor(neg_rels)
        neg_tri = (neg_h_samples, neg_t_samples, neg_rels)

        return pos_tri, neg_tri
            
    def __create_graph_data(self, id):
        edge_index = MOL_EDGE_LIST_FEAT_MTX[id][0]
        features = MOL_EDGE_LIST_FEAT_MTX[id][1]

        return Data(x=features, edge_index=edge_index)

    def __corrupt_ent(self, other_ent, r, other_ent_with_r_dict, max_num=1):
        corrupted_ents = []
        current_size = 0
        while current_size < max_num:
            candidates = np.random.choice(self.drug_ids, (max_num - current_size) * 2)
            mask = np.isin(candidates, other_ent_with_r_dict[(other_ent, r)], assume_unique=True, invert=True)
            corrupted_ents.append(candidates[mask])
            current_size += len(corrupted_ents[-1])
        
        if corrupted_ents != []:
            corrupted_ents = np.concatenate(corrupted_ents)

        return np.asarray(corrupted_ents[:max_num])
        
    def __corrupt_head(self, t, r, n=1):
        return self.__corrupt_ent(t, r, ALL_TRUE_H_WITH_TR, n)

    def __corrupt_tail(self, h, r, n=1):
        return self.__corrupt_ent(h, r, ALL_TRUE_T_WITH_HR, n)
    
    def __normal_batch(self, h, t, r, neg_size):
        neg_size_h = 0
        neg_size_t = 0
        prob = ALL_TAIL_PER_HEAD[r] / (ALL_TAIL_PER_HEAD[r] + ALL_HEAD_PER_TAIL[r])
        for i in range(neg_size):
            if random.random() < prob:
                neg_size_h += 1
            else:
                neg_size_t +=1
        
        return (self.__corrupt_head(t, r, neg_size_h),
                self.__corrupt_tail(h, r, neg_size_t))  


class DrugDataLoader(DataLoader):
    # Each epoch will generate random batches of data with the specified batch_size
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

