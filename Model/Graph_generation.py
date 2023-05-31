import torch
import os
import os.path as osp
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

class VPN(InMemoryDataset):
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # List of the raw files
        return ['edge.csv', 'graphid2label.csv', 'node2graphID.csv', 'nodeattrs.csv']
    
    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def download(self):
        pass
        
    def process(self):
        # Since we save as ".CSV" we read our files as panda data frame. 
        # DataFrames indices are adjusetd to start from 1
        path = os.path.join(self.raw_dir, 'nodeattrs.csv')
        node_attrs = pd.read_csv(path, sep=',', header=0)
        node_attrs.index += 1 

        path = os.path.join(self.raw_dir, 'edge.csv')
        edge_index = pd.read_csv(path, sep=',', header=0)
        edge_index.index += 1

        path = os.path.join(self.raw_dir, 'node2graphID.csv')
        nodes = pd.read_csv(path, sep=',', header=0)
        nodes.index += 1

        path = os.path.join(self.raw_dir, 'graphid2label.csv')
        graphID = pd.read_csv(path, sep=',', header=0)
        graphID.index += 1
        
        # Nodes attributes, edges connectivity and graph labels are extracted, 
        # The information is put in a data object and added to a list.
        data_list = []
        ids_list = nodes['graph_id'].unique()
        for graph_no in tqdm(ids_list):
            node_id = nodes.loc[nodes['graph_id']==graph_no].index
            
            attributes = node_attrs.loc[node_id, :]

            edges = edge_index.loc[edge_index['source_node'].isin(node_id)]
            edges_ids = edges.index

            label = graphID.loc[graph_no]
            
            edges_ids = torch.tensor(edges.to_numpy().transpose(), dtype=torch.long)
            map_dict = {v.item():i for i,v in enumerate(torch.unique(edges_ids))}
            map_edge = torch.zeros_like(edges_ids)
            for k,v in map_dict.items():
                map_edge[edges_ids==k] = v
                
            map_dict, map_edge, map_edge.shape
            
            edges_ids = map_edge.long()
            
            label = torch.tensor(label, dtype=torch.long)
            
            attrs = torch.tensor(attributes.to_numpy(),dtype=torch.float)
            
            graph = Data(x=attrs, edge_index=edges_ids, y=label)
            
            data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Process data is stored
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        
        
import torch_geometric.transforms as T
        
# Assign variable to stored data for easy accessibility
dataset = VPN(root='')
        
#To view graph parameters, graphs can be called with ids 
dataset[1]

z = dataset[1]  # Get the first graph object.

print()
print(f'Dataset: {dataset}:')
print('==============================')
print()
# Total num of graphs?
print(f'Number of graphs: {len(dataset)}')
# Num of features?
print(f'Number of features: {dataset.num_features}')
# Num of edges for graph 1?
print(f'Number of edges: {z.num_edges}')
# Number of labels?
print(f'Number of classes: {dataset.num_classes}')
# Average node degree?
print(f'Average node degree: {z.num_edges / z.num_nodes:.2f}')
# Do we have isolated nodes?
print(f'Contains isolated nodes: {z.has_isolated_nodes()}')
# Do we contain self-loops?
print(f'Contains self-loops: {z.has_self_loops()}')

#Visualize graph 1
networkx_graph = to_networkx(dataset[1])
plt.figure(1, figsize=(8,8))
networkx_graph.remove_nodes_from(list(nx.isolates(networkx_graph)))
nx.draw(networkx_graph, with_labels=True, node_size=300, font_size=10, font_color="black", font_weight="bold")
plt.show()
