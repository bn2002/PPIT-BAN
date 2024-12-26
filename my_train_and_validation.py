from sklearn import metrics
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
import  numpy as np
from torchdrug import data, utils
from torchdrug import layers
from my_args import *
import pickle
from torchdrug.models import GearNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix
from torch_geometric.nn import DataParallel

device = torch.device("cuda:0")

def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)




with open('./data/yeast/data_cpu.pickle', 'rb') as f:
    protein_data = pickle.load(f)

embed_data = np.load("./data/yeast/dictionary/protein_embeddings_esm2.npz")


def protein_transform(p1):  # 根据PDB文件生成蛋白质的图结构和序列数据
    protein_seq =[]
    protein_graph=[]
    protein_name =[]
    #proteins =[]
    for name in p1:

        if name[:3] =='gi:':
            name1 = name[3:]
        else:
            name1 = name

        node_number = embed_data[name].shape[0]
        # g_embed = torch.tensor(embed_data[name]).float().to(device)

        if node_number > 1200:
            # protein = protein[:1200]
            textembed = embed_data[name][:1200]
        else:
            textembed = np.concatenate((embed_data[name], np.zeros((1200 - node_number, 1280))))  # 1280,1024

        textembed = torch.tensor(textembed).float().to(device)
        # 取氨基酸序列
        protein_seq.append(textembed)
        protein_name.append(name)
        protein_graph.append(protein_data[name1].to(device))


        #proteins.append(protein)

       # print("===========:{}".format(torch.cuda.memory_allocated(0)))


    return protein_graph, protein_seq, protein_name


def predicting(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    #path = objectArgs['path']
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, (p1, p2, y) in enumerate(loader):

            proteins_graph1, proteins_seq1, protein_name1 = protein_transform(p1)
            proteins_graph2, proteins_seq2, protein_name2 = protein_transform(p2)

            output = model(proteins_graph1, proteins_seq1, proteins_graph2, proteins_seq2)
            output = torch.round(output.squeeze(1))
            total_preds = torch.cat((total_preds.cpu(), output.cpu()), 0)
            total_labels = torch.cat((total_labels.cpu(), y.float().cpu()), 0)
            
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


def train(trainArgs):

    train_losses = []
    train_accs = []


    for i in range(trainArgs['epochs']): 
        print("Running EPOCH",i+1)
        total_loss = 0
        n_batches = 0
        correct = 0
        train_loader = trainArgs['train_loader']
        optimizer = trainArgs['optimizer']
        criterion = trainArgs["criterion"]
        attention_model = trainArgs['model']
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)


        for batch_idx,(p1, p2, y) in enumerate(train_loader):

            proteins_graph1, proteins_seq1, protein_name1 = protein_transform(p1)

            proteins_graph2, proteins_seq2, protein_name2 = protein_transform(p2)

            y_pred = attention_model(proteins_graph1, proteins_seq1, proteins_graph2, proteins_seq2)
            correct+=torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),y.type(torch.DoubleTensor)).data.sum()
            loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))
            total_loss+=loss.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_batches+=1

            print(batch_idx)##########################################################

        avg_loss = total_loss/n_batches
        acc = correct.numpy()/(len(train_loader.dataset))


        
        train_losses.append(avg_loss)
        train_accs.append(acc)
        
        print("train avg_loss is", avg_loss)
        print("train ACC = ", acc)
        
        if(trainArgs['doSave']):
            torch.save(attention_model.state_dict(), pkl_path+'epoch'+'%d.pkl'%(i+1))
        # test
        total_labels,total_preds = predicting(attention_model, test_loader)
        test_acc = accuracy_score(total_labels, total_preds)
        test_prec = precision_score(total_labels, total_preds)
        test_recall = recall_score(total_labels, total_preds)
        test_f1 = f1_score(total_labels, total_preds)
        test_auc = roc_auc_score(total_labels, total_preds)
        con_matrix = confusion_matrix(total_labels, total_preds)
        test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
        test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
        print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
        with open(rst_file, 'a+') as fp:
            fp.write('epoch:' + str(i+1) + '\ttrainacc=' + str(acc) +'\ttrainloss=' + str(avg_loss.item()) +'\tacc=' + str(test_acc) + '\tprec=' + str(test_prec) + '\trecall=' + str(test_recall) +  '\tf1=' + str(test_f1) + '\tauc=' + str(test_auc) + '\tspec='+str(test_spec)+ '\tmcc='+str(test_mcc)+'\n')
    