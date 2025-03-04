# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:27:43 2024

@author: yxliao
"""


from PACCS.MolecularRepresentations import *
from PACCS.Training import *
import PACCS.Parameters as parameter

import pandas as pd
from pandas import DataFrame
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score
from numpy import mean, median, abs, sum, cumsum, histogram, sqrt
np.set_printoptions(suppress=True)


def metrics(y_true, y_pred):   #r2_score(y_true, y_pred)
    RelativeError = [abs(y_pred[i]-y_true[i])/y_true[i] for i in range(len(y_true))]
    R2_Score = r2_score(y_true, y_pred)
    abs_y_err = [abs(y_pred[i]-y_true[i]) for i in range(len(y_true))]
    mae = mean(abs_y_err)
    mdae = median(abs_y_err)
    MSE = mean_squared_error(y_true, y_pred)
    RMSE = sqrt(mean_squared_error(y_true, y_pred))
    print("R2 Score :", R2_Score)
    print("Mean Absolute Error :", mae)
    print("Median Absolute Error :", mdae)
    print("Median Relative Error :", np.median(RelativeError) * 100, '%')
    print("Mean Relative Error :", np.mean(RelativeError) * 100, '%')
    print("Root Mean Squared Error :", RMSE)
    print("Mean Squared Error :", MSE)
    return R2_Score, np.median(RelativeError) * 100


def PACCS_predict(input_path, model_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    smiles, adduct, ccs, vpa, mz = input_data(input_path)
    smiles2 = smiles
    adduct2 = adduct
    
    print('## Read data : ',len(smiles))
    print('## All Atoms : ', parameter.All_Atoms)
    print('## All Adduct types: ', parameter.Adduct_SET)
 
    adduct_one_hot = [(one_of_k_encoding_unk(adduct[_], parameter.Adduct_SET)) for _ in range(len(adduct))]
    adduct_one_hot = list(np.array(adduct_one_hot).astype(int))
    
    xmin = parameter.Min_vpa
    xmax = parameter.Max_vpa
    for i, x in enumerate(vpa):
        vpa[i] = (x-xmin) / (xmax-xmin)
    
    wmin = parameter.Min_mz
    wmax = parameter.Max_mz
    for j, m in enumerate(mz):
        mz[j] = (m-wmin) / (wmax-wmin)
    
    print("Test length :", len(vpa))
    
    graph_adduct_data = load_representations(smiles, adduct_one_hot, ccs, vpa, mz)
    
    test_loader = torch.utils.data.DataLoader(
        graph_adduct_data,
        shuffle=False,
        num_workers=0,
        batch_size=1
    )
    
    pred_ccs = []
    true_ccs = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            graph = Data(
                x=data['x'][0], 
                edge_index=data['edge_index'][0],
                edge_attr=data['edge_attr'][0], 
                y=data['y'][0]
            ).to(device)
        
            adduct = data['adduct'][0].to(device)
            vpa = data['vpa'][0].to(device)
            mz = data['mz'][0].to(device)
    
            m_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            new_m = PACCS().to(device)
            new_m.load_state_dict(m_state_dict)
            predict_test = new_m(graph, vpa, mz, adduct)
            pred_ccs.append(predict_test[0].cpu().numpy().tolist())
            true_ccs.append(data['y'][0][0].numpy().tolist())
            
    if len(smiles2) == len(adduct2) == len(true_ccs) == len(pred_ccs):
        data2 = {
            'SMILES': smiles2,
            'Adduct': adduct2,
            'True CCS': true_ccs,
            'Predicted CCS': pred_ccs
        }
        df = DataFrame(data2)
        df.to_csv(output_path, index=False)
        print('## CCS prediction has been completed')
    else:
        print("Error: Length mismatch!")
        print(f"Lengths - SMILES: {len(smiles2)}, Adduct: {len(adduct2)}, True CCS: {len(true_ccs)}, Predicted CCS: {len(pred_ccs)}")

    re_Metrics = metrics(true_ccs, pred_ccs)


def PACCS_predict_woeccs(input_path, model_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    smiles, adduct, vpa, mz = input_data_woeccs(input_path)
    smiles2 = smiles
    adduct2 = adduct
    vpa2 = vpa.copy()
    mz2 = mz.copy()
    
    print('## Read data : ',len(smiles))
    print('## All Atoms : ', parameter.All_Atoms)
    print('## All Adduct types: ', parameter.Adduct_SET)
 
    adduct_one_hot = [(one_of_k_encoding_unk(adduct[_], parameter.Adduct_SET)) for _ in range(len(adduct))]
    adduct_one_hot = list(np.array(adduct_one_hot).astype(int))
    
    xmin = parameter.Min_vpa
    xmax = parameter.Max_vpa
    for i, x in enumerate(vpa):
        vpa[i] = (x-xmin) / (xmax-xmin)
    
    wmin = parameter.Min_mz
    wmax = parameter.Max_mz
    for j, m in enumerate(mz):
        mz[j] = (m-wmin) / (wmax-wmin)
    
    print("Test length :", len(vpa))
    
    graph_adduct_data = load_representations_woeccs(smiles, adduct_one_hot, vpa, mz)
    
    
    test_loader = torch.utils.data.DataLoader(
        graph_adduct_data,
        shuffle=False,
        num_workers=0,
        batch_size=1
    )
    
    pred_ccs = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            graph = Data(
                x=data['x'][0], 
                edge_index=data['edge_index'][0],
                edge_attr=data['edge_attr'][0]
            ).to(device)
        
            adduct = data['adduct'][0].to(device)
            vpa = data['vpa'][0].to(device)
            mz = data['mz'][0].to(device)
    
            m_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            new_m = PACCS().to(device)
            new_m.load_state_dict(m_state_dict)
            predict_test = new_m(graph, vpa, mz, adduct)
            pred_ccs.append(predict_test[0].cpu().numpy().tolist())
            
    if len(smiles2) == len(adduct2) == len(vpa2) == len(mz2) == len(pred_ccs):
        data2 = {
            'SMILES': smiles2,
            'Adduct': adduct2,
            'vpa': vpa2,
            'mz': mz2,
            'Predicted CCS': pred_ccs
        }
        df = DataFrame(data2)
        df.to_csv(output_path, index=False)
        print('## CCS prediction has been completed')
    else:
        print("Error: Length mismatch!")
        print(f"Lengths - SMILES: {len(smiles2)}, Adduct: {len(adduct2)}, vpa: {len(vpa2)}, mz: {len(mz2)}, Predicted CCS: {len(pred_ccs)}")


def PACCS_predict_blengths(input_path, model_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    smiles, adduct, ccs, vpa, mz, blengths = input_data_blengths(input_path)
    
    smiles2 = smiles
    adduct2 = adduct
    
    print('## Read data : ',len(smiles))
    print('## All Atoms : ', parameter.All_Atoms)
    print('## All Adduct types: ', parameter.Adduct_SET)
 
    adduct_one_hot = [(one_of_k_encoding_unk(adduct[_], parameter.Adduct_SET)) for _ in range(len(adduct))]
    adduct_one_hot = list(np.array(adduct_one_hot).astype(int))
    
    xmin = parameter.Min_vpa
    xmax = parameter.Max_vpa
    for i, x in enumerate(vpa):
        vpa[i] = (x-xmin) / (xmax-xmin)
    
    wmin = parameter.Min_mz
    wmax = parameter.Max_mz
    for j, m in enumerate(mz):
        mz[j] = (m-wmin) / (wmax-wmin)
    
    print("Test length :", len(vpa))
    
    graph_adduct_data = load_representations_blengths(smiles, adduct_one_hot, ccs, vpa, mz, blengths)
    
    test_loader = torch.utils.data.DataLoader(
        graph_adduct_data,
        shuffle=False,
        num_workers=0,
        batch_size=1
    )
    
    pred_ccs = []
    true_ccs = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            graph = Data(
                x=data['x'][0], 
                edge_index=data['edge_index'][0],
                edge_attr=data['edge_attr'][0], 
                y=data['y'][0]
            ).to(device)
        
            adduct = data['adduct'][0].to(device)
            vpa = data['vpa'][0].to(device)
            mz = data['mz'][0].to(device)
    
            m_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            new_m = PACCS().to(device)
            new_m.load_state_dict(m_state_dict)
            predict_test = new_m(graph, vpa, mz, adduct)
            pred_ccs.append(predict_test[0].cpu().numpy().tolist())
            true_ccs.append(data['y'][0][0].numpy().tolist())
            
    if len(smiles2) == len(adduct2) == len(true_ccs) == len(pred_ccs):
        data2 = {
            'SMILES': smiles2,
            'Adduct': adduct2,
            'True CCS': true_ccs,
            'Predicted CCS': pred_ccs
        }
        df = DataFrame(data2)
        df.to_csv(output_path, index=False)
        print('## CCS prediction has been completed')
    else:
        print("Error: Length mismatch!")
        print(f"Lengths - SMILES: {len(smiles2)}, Adduct: {len(adduct2)}, True CCS: {len(true_ccs)}, Predicted CCS: {len(pred_ccs)}")

    re_Metrics = metrics(true_ccs, pred_ccs)
