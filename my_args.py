from dataset_loader import *
from model import *

device = torch.device('cuda', 0)
modelArgs = {}
modelArgs['bcn_heads'] = 1                  #####head
modelArgs['batch_size'] = 32
modelArgs['dropout'] = 0.5
modelArgs['emb_dim'] = 1280                          #1280,1024
modelArgs['output_dim'] = 512
#modelArgs['dense_hid'] = 64
modelArgs['task_type'] = 0
modelArgs['n_classes'] = 1


trainArgs = {}
trainArgs['model'] = PPI_BAN(modelArgs).to(device)
trainArgs['epochs'] = 60
trainArgs['lr'] = 0.0003
trainArgs['train_loader'] = train_loader
trainArgs['doTest'] = True
trainArgs['criterion'] = torch.nn.BCELoss()
trainArgs['optimizer'] = torch.optim.AdamW(trainArgs['model'].parameters(), lr=trainArgs['lr'], weight_decay=0.001)
trainArgs['doSave'] = False

