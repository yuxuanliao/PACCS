# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 01:29:45 2024

@author: yxliao
"""


from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.Descriptors import ExactMolWt
from tqdm import tqdm

import numpy as np
import pandas as pd
from pandas import *


def SmilesMW(smiles, adduct):
    MW = []
    for i in tqdm(range(len(smiles))):
        smi = smiles[i]
        add = adduct[i]
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        cal_mass = ExactMolWt(mol)
        if add == '[M+Na]+':
            cal_mass += ExactMolWt(Chem.MolFromSmiles('[Na+]'))
        elif add == '[M+H]+':
            cal_mass += ExactMolWt(Chem.MolFromSmiles('[H+]'))
        elif add == '[M-H]-':
            cal_mass -= ExactMolWt(Chem.MolFromSmiles('[H+]'))
        MW.append(cal_mass)
    return MW


