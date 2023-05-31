#Building the GNN

from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.loader import DataLoader, ImbalancedSampler
from torch_geometric.nn import GCNConv, GATConv, TopKPooling, BatchNorm, GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.utils.data as data
from torchvision import datasets
from torch.utils.data import WeightedRandomSampler

#Shuffle the dataset
dataset = dataset.shuffle()

#Split the dataset into train and test
train_dataset = dataset [:13886]
test_dataset = dataset [13886:16337]

#Review the training versus test data
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

#Define sampler and data loaders
sampler = ImbalancedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

#View train and test classes to ensure all classes are represented 
print(f'Number of training classes: {train_dataset.num_classes}')
print(f'Number of test classes : {test_dataset.num_classes}')

# Define our GCN class as a pytorch Module
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.5)
        self.bn1 = BatchNorm1d(128)
        
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.5)
        self.bn2 = BatchNorm1d(128)
        
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.5)
        self.bn3 = BatchNorm1d(128)
        
        self.conv4 = GraphConv(128, 128)
        self.pool4 = TopKPooling(128, ratio=0.5)
        self.bn4 = BatchNorm1d(128)
        
        self.conv5 = GraphConv(128, 128)
        self.pool5 = TopKPooling(128, ratio=0.5)
        self.bn5 = BatchNorm1d(128)
        
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64 )
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)         
        
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x, edge_index, _, batch, _, _ = self.pool5(x, edge_index, None, batch)
        x5 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) 
       
        x = x1+x2+x3+x4+x5  #+x10+x11+x12+x13   #+x6+x7+x8+x9
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)
        
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN()
model.cuda()
print(model)
