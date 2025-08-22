from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix
from my_args import *
import pickle
import  numpy as np
from torch.utils.data import Subset

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
            print("Processing batch {}".format(batch_idx))
            proteins_graph1, proteins_seq1, protein_name1 = protein_transform(p1)
            proteins_graph2, proteins_seq2, protein_name2 = protein_transform(p2)

            output = model(proteins_graph1, proteins_seq1, proteins_graph2, proteins_seq2)
            output = torch.round(output.squeeze(1))
            total_preds = torch.cat((total_preds.cpu(), output.cpu()), 0)
            total_labels = torch.cat((total_labels.cpu(), y.float().cpu()), 0)
            
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


attention_model = trainArgs['model']
attention_model.load_state_dict(torch.load('model_pkl/GATepoch.pkl', map_location='cpu'))
attention_model.eval()

# Tạo subset nhỏ chỉ có 32 samples (đủ 1 batch)
small_test_dataset = Subset(test_dataset, range(256))
small_test_loader = DataLoader(dataset = small_test_dataset, batch_size = 32, shuffle=False, drop_last = True, collate_fn=collate)

total_labels,total_preds = predicting(attention_model, small_test_loader)
test_acc = accuracy_score(total_labels, total_preds)
print("acc: ",test_acc)
print("Labels:", total_labels)
print("Predictions:", total_preds)