import os
import pickle
from torchdrug import data, layers



file_dir = './data/yeast/PDB'
protein_data = {}
files = os.listdir(file_dir)
for pdb_file in files:
    name = pdb_file[:-4]
    protein = data.Protein.from_pdb(file_dir + '/' + pdb_file, atom_feature="position", bond_feature="length",
                                    residue_feature="symbol")
    if protein.num_residue > 1200:
        protein = protein[:1200]
    _protein = data.Protein.pack([protein])

    graph_construction_model = layers.GraphConstruction(node_layers=[layers.geometry.AlphaCarbonNode()],
                                                            edge_layers=[
                                                                layers.geometry.SpatialEdge(radius=10.0,
                                                                                            min_distance=5),
                                                                layers.geometry.KNNEdge(k=10, min_distance=5),
                                                                layers.geometry.SequentialEdge(max_distance=2)],
                                                            edge_feature="gearnet")

    protein_data[name] = graph_construction_model(_protein)

with open('data_cpu.pickle', 'wb') as f:
    pickle.dump(protein_data, f)