import torch
import scipy.sparse as spp
from seq2tensor import s2t 
import os
import numpy as np
import re
import sys
from torch.utils.data import DataLoader,Dataset
from torch_geometric.loader import DataListLoader
import sys


device = torch.device("cuda:0")
if len(sys.argv) > 1:
    datasetname, rst_file, pkl_path, batchsize = sys.argv[1:]
    batchsize = int(batchsize)
else:
    datasetname = 'yeast'            #'yeast'
    actions = '0'
    fold = '0'
    rst_file = './results/yeast_pipr.tsv'
    pkl_path = './model_pkl/GAT'
    batchsize = 32


def collate(samples):

    p1,p2,labels = map(list, zip(*samples))
    return p1,p2,torch.tensor(labels)


class MyDataset(Dataset): 

    def __init__(self,type,transform=None,target_transform=None):
        
        super(MyDataset,self).__init__()
        pns=[]
        with open('./data/'+datasetname+'/'+'actions'+'/'+actions+'/'+type +"_"+fold+'.actions.tsv', 'r') as fh:
            for line in fh: 
                line = line.strip('\n')
                line = line.rstrip('\n')
                words = re.split('  |\t',line)
                pns.append((words[0],words[1],int(words[2])))
                
        self.pns = pns
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        p1,p2, label = self.pns[index]

        return p1,p2, label

       
    def __len__(self):
        return len(self.pns)


train_dataset = MyDataset(type = 'train')
test_dataset = MyDataset(type = 'test')

train_loader = DataLoader(dataset = train_dataset, batch_size = batchsize, shuffle=True,drop_last = True,collate_fn=collate,generator=torch.Generator(device = device))
test_loader = DataLoader(dataset = test_dataset, batch_size = batchsize , shuffle=True,drop_last = True,collate_fn=collate,generator=torch.Generator(device = device))

