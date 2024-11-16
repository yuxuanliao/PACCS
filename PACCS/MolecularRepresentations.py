# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 00:22:05 2024

@author: yxliao
"""

import pandas as pd
import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdchem import BondType as BT

from tqdm import tqdm
import PACCS.Parameters as parameter
from multiprocessing.pool import ThreadPool

from PACCS.VoxelProjectedArea import *
from PACCS.MZ import *


def input_data(filename,):
    data = pd.read_csv(filename)
    smiles = list(data['SMILES'])
    adduct = list(data['Adduct'])
    ccs    = list(data['True CCS'])   
    if 'vpa' in data.columns:
        vpa = list(data['vpa'])
    else:
        pool = ThreadPool(16)
        re = pool.map(smilesPA, smiles)
        pool.close()
        pool.join()
        vpa = np.mean(re,axis=1)
        data['vpa'] = vpa
        data.to_csv('./Data/input_data_vpa.csv', index=False)
    if 'mz' in data.columns:
        mz = list(data['mz'])
    else:
        mz = SmilesMW(smiles, adduct)
        data['mz'] = mz
        data.to_csv('./Data/input_data_mz.csv', index=False)

    return smiles, adduct, ccs, vpa, mz


def Standardization(data):
    data_list = [data[i] for i in data]
    Max_data, Min_data = np.max(data_list), np.min(data_list)
    for i in data:
        data[i] = (data[i] - Min_data) / (Max_data - Min_data)
    return data

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_feature_oneHot(atom, All_Atoms, Atom_radius, Atom_mass):
    return np.array(
        # Atomic Type (One-Hot)
        one_of_k_encoding_unk(atom.GetSymbol() ,All_Atoms) +
        # Atomic Degree (One-Hot)
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4]) +
        # Atomic radius  Atomic mass (float)
        [Atom_radius[atom.GetSymbol()],Atom_mass[atom.GetSymbol()]] +
        # Atomic is in Ring ? (One-Hot)
        one_of_k_encoding_unk(atom.IsInRing(), [0, 1])
    )


def smiles2Graph(smi):
    Atom_radius = Standardization(parameter.Atom_radius)
    Atom_mass = Standardization(parameter.Atom_mass)   
    
    m = Chem.MolFromSmiles(smi)
    mol = Chem.RemoveHs(m)
    mol3D = Chem.AddHs(m)

    ps = AllChem.ETKDGv3()
    ps.randomSeed = -1
    ps.maxAttempts = 1
    ps.numThreads = 0
    ps.useRandomCoords = True
    re = AllChem.EmbedMultipleConfs(mol3D, numConfs = 1, params = ps)
    re = AllChem.MMFFOptimizeMoleculeConfs(mol3D, numThreads = 0)
    conf = mol3D.GetConformer()

    N = mol.GetNumAtoms()

    x = []
    for atom in mol.GetAtoms():
        x.append(atom_feature_oneHot(atom, parameter.All_Atoms, Atom_radius, Atom_mass))
    
    row, col, edge_attr = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        
        bond_length = rdMolTransforms.GetBondLength(conf, start, end)
        
        edge_attr += 2 * [one_of_k_encoding_unk(bond.GetBondTypeAsDouble(), [1, 1.5, 2, 3]) + [bond_length]]
        
    x = torch.tensor(np.array(x), dtype=torch.float32)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    
    return x, edge_attr, edge_index

def load_representations(smiles, adduct_one_hot, ccs, vpa, mz):
    graph_adduct_data = []
    Index = 0
    for smi in tqdm(smiles):
        
        g_x, g_e, g_i = smiles2Graph(smi)
    
        # label: true CCS
        y = torch.tensor([ccs[Index]], dtype=torch.float)
        
        one_graph = {}
        one_graph['x'] = g_x
        one_graph['edge_index'] = g_i
        one_graph['edge_attr'] = g_e
        
        one_graph['y'] = y
        one_graph['adduct'] = adduct_one_hot[Index]
        one_graph['vpa'] = torch.tensor([vpa[Index]], dtype=torch.float)
        one_graph['mz'] = torch.tensor([mz[Index]], dtype=torch.float)
        graph_adduct_data.append(one_graph)
        
        Index += 1
    return graph_adduct_data

