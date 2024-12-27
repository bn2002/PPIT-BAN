# PPIT-BAN

This repository comprises the source code, datasets, and models associated with our research paper entitled: 
Predicting protein-protein interaction with interpretable bilinear attention network.

# Dependencies
* python = 3.8
* pytorch = 1.10.0
* torchdrug = 0.2.1
* transformers = 4.28.1
* numpy
* pandas
* sklearn

# Datasets
The datasets we used in the study can be downloaded through the following links.

* [Yeast dataset]( https://pan.baidu.com/s/1G2vMODUVNlMyucRXqRz--w?pwd=191c)      password: 191c
* [Multi-species dataset]( https://pan.baidu.com/s/1PrTJdJ4TOLSzhWclIcqlUA?pwd=au29)      password: au29
* [Multi-class dataset]( https://pan.baidu.com/s/1_GUXsFQIZ24iP002z3StPg?pwd=jdbt)      password: jdbt


# Run 
After prepared with suitable environment, you can execute the following .py files with default parameters.

1.First, execute the code below to generate embeddings for the protein sequences.
```shell
protein_sequence_embedding.py
```
2.Next, execute the code below to generate structural data for the proteins.
```shell
protein_graph_establish.py
```
3.Finally, implement downstream tasks by running the code below.
```shell
my_main.py
```

