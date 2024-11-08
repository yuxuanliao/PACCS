# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 01:27:55 2024

@author: yxliao
"""

import rdkit
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import numpy as np

import PACCS.Parameters as parameter




def fibonacci_sphere(numpts: int):
    ga = (3 - np.sqrt(5)) * np.pi # golden angle
    theta = ga * np.arange(numpts)
    z = np.linspace(1/numpts-1, 1-1/numpts, numpts)
    radius = np.sqrt(1 - z * z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack((x,y,z), axis=1)


def smilesPA(smi, Num=10, cell_radii=0.5, sample_num=100):
    PA = []
    for _ in range(Num):
        iMol = Chem.MolFromSmiles(smi)
        iMol3D = Chem.AddHs(iMol)
    
        ps = AllChem.ETKDGv3()
        ps.randomSeed = -1
        ps.maxAttempts = 1
        ps.numThreads = 0
        ps.useRandomCoords = True
        re = AllChem.EmbedMultipleConfs(iMol3D, numConfs = 1, params = ps)
        re = AllChem.MMFFOptimizeMoleculeConfs(iMol3D,  numThreads = 0)

        Coords = []
        pos = fibonacci_sphere(sample_num)
        
        for atom in iMol3D.GetAtoms():
            Symbol = atom.GetSymbol()
            Coord = list(iMol3D.GetConformer().GetAtomPosition(atom.GetIdx()))
            Coords.append(np.array(Coord) + pos * parameter.atomic_radii[Symbol])

        Coords = np.vstack(Coords)
        grid = (np.array(Coords) // cell_radii).astype(int)
        grid = np.unique(grid, axis=0)

        p_x = [(i[1],i[2]) for i in grid]
        p_y = [(i[0],i[2]) for i in grid]
        p_z = [(i[0],i[1]) for i in grid]

        pa = (len(set(p_x)) + len(set(p_y)) + len(set(p_z))) / 3
        PA.append(pa)
    return PA