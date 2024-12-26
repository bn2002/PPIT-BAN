
from transformers import BertTokenizer, AutoTokenizer,BertModel, AutoModel
import pandas as pd
import numpy as np

tokenizer_ = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model_ = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

list_data={}
#input prtotein raw sequnences
protein_list = pd.read_csv('./data/yeast/dictionary/protein.dictionary.tsv',sep="\t", header=None)
file_path = './data/yeast/dictionary/protein_embeddings_esm2.npz'



for i in range(len(protein_list)):
    protein_name = protein_list.iloc[i, 0]
    protein_seq = protein_list.iloc[i, 1]
    outputs_ = model_(**tokenizer_(protein_seq, return_tensors='pt'))
    protein_embed = outputs_.last_hidden_state[:, 1:-1, :]
    protein_embed = protein_embed.sum(axis=0).detach().numpy()
    list_data.update({protein_name: protein_embed})
    print(i)

np.savez(file_path, **list_data)