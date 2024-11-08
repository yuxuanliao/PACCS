# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 00:48:24 2024

@author: yxliao
"""


import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import NNConv
from PACCS.MolecularRepresentations import *


class PACCS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        '''Graph 卷积'''
        self.lin0 = torch.nn.Linear(20, 64)
        nn = Sequential(Linear(5, 64), ReLU(), Linear(64, 64 * 64))
        self.conv = NNConv(64, 64, nn, aggr='mean')
        #self.set2set = Set2Set(dim, processing_steps = 3)  ###
        self.graph_conv_N = 3
        
        '''MLP'''
        self.bottelneck = torch.nn.Linear(64 + 2 + 3, 384)
        self.lin1 = torch.nn.Linear(384, 384)
        self.lin2 = torch.nn.Linear(384, 1)
        self.MLP_N = 6

    def forward(self, graph, vpa, mz, adduct):
        out_g = F.relu(self.lin0(graph.x))
        for i in range(self.graph_conv_N):
            out_g = F.relu(self.conv(out_g, graph.edge_index, graph.edge_attr))
            out_g = out_g.squeeze(0)            
        out_g = torch.sum(out_g, dim=0)
        
        out = torch.cat([out_g, vpa, mz, adduct], dim=-1)
        #out = self.set2set(out)
        out = F.relu(self.bottelneck(out))
        for j in range(self.MLP_N):
            out = F.relu(self.lin1(out))
        out = self.lin2(out)

        return out
    
def PACCS_train(input_path, epochs, batchsize, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    smiles, adduct, ccs, vpa, mz = input_data(input_path)
    print('## Read data : ',len(smiles))
    
    adduct_SET = list(set(adduct))
    adduct_SET.sort()
    print('## Adduct set order : ', adduct_SET)    
    adduct_one_hot = [(one_of_k_encoding_unk(adduct[_], adduct_SET)) for _ in range(len(adduct))]
    adduct_one_hot = list(np.array(adduct_one_hot).astype(int))
    
    graph_adduct_data = load_representations(smiles, adduct_one_hot, ccs, vpa, mz)
    split_line1 = int(len(ccs) - len(ccs)*0.2)
    split_line2 = int(len(ccs) - len(ccs)*0.2) + int(len(ccs) * 0.1)
    
    train_adduct_data = graph_adduct_data[:split_line1]
    valid_adduct_data = graph_adduct_data[split_line1:split_line2]
    test_adduct_data = graph_adduct_data[split_line2:]
    
    
    print('## The size of the training set : ', len(train_adduct_data))
    print('## The size of the validation set : ', len(valid_adduct_data))

    train_loader = torch.utils.data.DataLoader(
        train_adduct_data,
        shuffle=True,
        num_workers=0,
        batch_size=1
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_adduct_data,
        shuffle=False,
        num_workers=0,
        batch_size=1
    )
    
    model = PACCS().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for _ in range(1, epochs+1):
        batch_loss = 0
        Index = 0
        
        loss_all = []
        with tqdm(total=len(train_loader)) as p_bar:
            for data in train_loader:
                Index += 1
    
                graph = Data(
                    x=data['x'][0], 
                    edge_index=data['edge_index'][0],
                    edge_attr=data['edge_attr'][0], 
                    y=data['y'][0]
                ).to(device)
    
                adduct = data['adduct'][0].to(device)
                vpa = data['vpa'][0].to(device)
                mz = data['mz'][0].to(device)
    
                pred = model(graph, vpa, mz, adduct)
    #             loss = F.mse_loss(pred, graph.y)
                loss = F.huber_loss(pred, graph.y)
                loss_all.append(loss.cpu().detach().numpy())
    
                batch_loss += loss
                if Index % batchsize == 0:
                    optimizer.zero_grad()
                    batch_loss = batch_loss / batchsize
                    batch_loss.backward()
                    optimizer.step()
                    
                    p_bar.update(batchsize)
                    p_bar.set_description("Training-Loss {:.2f}".format(batch_loss))
                    batch_loss = 0
        train_loss = np.mean(loss_all)
            
        loss_all = []
        with torch.no_grad():
            for data in tqdm(valid_loader):
                graph = Data(
                    x=data['x'][0], 
                    edge_index=data['edge_index'][0],
                    edge_attr=data['edge_attr'][0], 
                    y=data['y'][0]
                ).to(device)
            
                adduct = data['adduct'][0].to(device)
                vpa = data['vpa'][0].to(device)
                mz = data['mz'][0].to(device)
    
                pred = model(graph, vpa, mz, adduct)
                
                # pred = model(graph, line_graph, adduct)
                # loss = F.mse_loss(pred, graph.y)
                loss = F.huber_loss(pred, graph.y)
                loss_all.append(loss.cpu().detach().numpy())
        val_loss = np.mean(loss_all)
        # scheduler.step(val_loss)
        
        print('train-loss', train_loss, 'val-loss', val_loss)

        torch.save(model.state_dict(), output_path)