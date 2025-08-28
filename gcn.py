import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

class TwoGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        for m in self.modules():
            if isinstance(m, GCNConv):
                torch.nn.init.xavier_uniform_(m.lin.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index):
        edge_index = edge_index.cuda()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, num_class=8, h_dim=49, g_dim=1024):
        super(GCN, self).__init__()
        self.gcn_num = TwoGCN(num_class, num_class, num_class)
        self.gcn_h = TwoGCN(h_dim, h_dim, h_dim)
        self.gcn_g = TwoGCN(g_dim, g_dim, 128)

        self.edge_index_g = dense_to_sparse(torch.ones(num_class*h_dim, num_class*h_dim))[0]
        self.edge_index_h = dense_to_sparse(torch.ones(num_class * 128, num_class * 128))[0]
        self.edge_index_num = dense_to_sparse(torch.ones(h_dim * 128, h_dim * 128))[0]

    
    def forward(self, x):
        data_list = []
        b, c, d, l = x.shape
        l_out = 128
        num_nodes = c * d
        for i in range(b):
            x_i = x[i].view(num_nodes, l) 
            
            edge_index = self.edge_index_g  
    
            data = Data(x=x_i, edge_index=edge_index)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        out = self.gcn_g(batch.x, batch.edge_index)

        out_list = out.split(num_nodes, dim=0)  # List of [c*d, l_out]

        x = torch.stack([x_i.view(c, d, l_out) for x_i in out_list], dim=0)
    
        x = x.transpose(2, 3)  
        data_list = []
        b, c, d, l = x.shape
        l_out = 49
        num_nodes = c * d
        for i in range(b):
            x_i = x[i].reshape(num_nodes, l)  
    
            edge_index = self.edge_index_h 
    
            data = Data(x=x_i, edge_index=edge_index)
            data_list.append(data)
            
        batch = Batch.from_data_list(data_list)
        out = self.gcn_h(batch.x, batch.edge_index)

        out_list = out.split(num_nodes, dim=0)  

        x = torch.stack([x_i.view(c, d, l_out) for x_i in out_list], dim=0) 
        x = x.transpose(2, 3) 
        x = x.transpose(1,3) 
        data_list = []
        b, c, d, l = x.shape
        l_out = 8
        num_nodes = c * d
        for i in range(b):
            x_i = x[i].reshape(num_nodes, l) 
    
            edge_index = self.edge_index_num  
    
            data = Data(x=x_i, edge_index=edge_index)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        out = self.gcn_num(batch.x, batch.edge_index)

        out_list = out.split(num_nodes, dim=0)  

        x = torch.stack([x_i.view(c, d, l_out) for x_i in out_list], dim=0) 
        x = x.transpose(1, 3) 
    
        return x







