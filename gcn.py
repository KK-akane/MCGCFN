import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

# 定义 GCN 模型
class TwoGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # 初始化 GCNConv 层的权重
        for m in self.modules():
            if isinstance(m, GCNConv):
                torch.nn.init.xavier_uniform_(m.lin.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index):
        edge_index = edge_index.cuda()
        # 第一层 GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层 GCN
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


# torch.Size([1, 13, 49, 128])
class GCN(torch.nn.Module):
    def __init__(self, num_class=8, h_dim=49, g_dim=1024):
        super(GCN, self).__init__()
        self.gcn_num = TwoGCN(num_class, num_class, num_class)
        self.gcn_h = TwoGCN(h_dim, h_dim, h_dim)
        self.gcn_g = TwoGCN(g_dim, g_dim, 128)
        # GCN中加入以下构图逻辑
        self.edge_index_g = dense_to_sparse(torch.ones(num_class*h_dim, num_class*h_dim))[0]
        self.edge_index_h = dense_to_sparse(torch.ones(num_class * 128, num_class * 128))[0]
        self.edge_index_num = dense_to_sparse(torch.ones(h_dim * 128, h_dim * 128))[0]

    # def forward(self, x):
        # x = x.squeeze(0)
        # # 128
        # c,d,l = x.shape
        # # 转为 [c*d, l]
        # x_nodes = x.view(c * d, l)
        # # 构建全连接图（含自环）
        # # adj = torch.ones(c * d, c * d)
        # # edge_index, _ = dense_to_sparse(adj)
        # x = self.gcn_g(x_nodes, self.edge_index_g)
        # x = x.view(c, d, int(l/8)) # [13,49,128]

        # # 49维度
        # x = x.transpose(1, 2) # [13,128,49]
        # c, d, l = x.shape
        # x_nodes = x.reshape(c * d, l)
        # # adj = torch.ones(c * d, c * d)
        # # edge_index, _ = dense_to_sparse(adj)
        # x = self.gcn_h(x_nodes, self.edge_index_h)
        # x = x.reshape(c, d, l)
        # x = x.transpose(1, 2) # [13,49,128]

        # # 13
        # x = x.transpose(0, 2) # [128,49,13]
        # c, d, l = x.shape
        # x_nodes = x.reshape(c * d, l)
        # # adj = torch.ones(c * d, c * d)
        # # edge_index, _ = dense_to_sparse(adj)
        # x = self.gcn_num(x_nodes, self.edge_index_num)
        # x = x.reshape(c, d, l)
        # x = x.transpose(0, 2) # [13,49,128]

        # x = x.unsqueeze(0)

        # return x




    # batch
    def forward(self, x):
        # 512
        # 构建 batch 图
        data_list = []
        b, c, d, l = x.shape
        l_out = 128
        # 每个图节点数
        num_nodes = c * d
        for i in range(b):
            x_i = x[i].view(num_nodes, l)  # [c*d, l]
    
            # 全连接图的邻接矩阵
            edge_index = self.edge_index_g  # [2, num_edges]
    
            data = Data(x=x_i, edge_index=edge_index)
            data_list.append(data)
        # 拼成 batch
        batch = Batch.from_data_list(data_list)
        out = self.gcn_g(batch.x, batch.edge_index)
        # 拆分成每个样本
        out_list = out.split(num_nodes, dim=0)  # List of [c*d, l_out]
        # 还原成 [b, c, d, l_out]
        x = torch.stack([x_i.view(c, d, l_out) for x_i in out_list], dim=0) # [b,13,49,128]
    
        # 49维度
        x = x.transpose(2, 3)   # [b,13,128,49]
        data_list = []
        b, c, d, l = x.shape
        l_out = 49
        # 每个图节点数
        num_nodes = c * d
        for i in range(b):
            x_i = x[i].reshape(num_nodes, l)  # [c*d, l]
    
            # 全连接图的邻接矩阵
            edge_index = self.edge_index_h  # [2, num_edges]
    
            data = Data(x=x_i, edge_index=edge_index)
            data_list.append(data)
        # 拼成 batch
        batch = Batch.from_data_list(data_list)
        out = self.gcn_h(batch.x, batch.edge_index)
        # 拆分成每个样本
        out_list = out.split(num_nodes, dim=0)  # List of [c*d, l_out]
        # 还原成 [b, c, d, l_out]
        x = torch.stack([x_i.view(c, d, l_out) for x_i in out_list], dim=0) # [b,13,128,49]
        x = x.transpose(2, 3) # [b,13,49,128]
    
        # num_class
        x = x.transpose(1,3) # [b,128,49,13]
        data_list = []
        b, c, d, l = x.shape
        l_out = 8
        # 每个图节点数
        num_nodes = c * d
        for i in range(b):
            x_i = x[i].reshape(num_nodes, l)  # [c*d, l]
    
            # 全连接图的邻接矩阵
            edge_index = self.edge_index_num  # [2, num_edges]
    
            data = Data(x=x_i, edge_index=edge_index)
            data_list.append(data)
        # 拼成 batch
        batch = Batch.from_data_list(data_list)
        out = self.gcn_num(batch.x, batch.edge_index)
        # 拆分成每个样本
        out_list = out.split(num_nodes, dim=0)  # List of [c*d, l_out]
        # 还原成 [b, c, d, l_out]
        x = torch.stack([x_i.view(c, d, l_out) for x_i in out_list], dim=0) # [b,128,49,13]
        x = x.transpose(1, 3) # [b,13,49,128]
    
        return x







# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GCN, self).__init__()
#         self.gcn = SimpleGCN(input_dim, hidden_dim, output_dim)
#
#     def forward(self, x):
#         # 13的维度
#         edge_index_13 = fully_connected_edge_index(8)
#         reshaped_data = x.permute(0, 2, 3, 1).reshape(-1, 1)  # [batchsize * 49 * 2048 * 13, 1]
#         # 扩展邻接矩阵到所有图上
#         num_graphs = x.shape[0] * 49 * 512  # 图的数量
#         num_nodes = 8  # 每个图的节点数
#         edge_index_expanded = []
#         for i in range(num_graphs):
#             offset = i * num_nodes
#             edge_index_expanded.append(edge_index_13 + offset)
#         edge_index_expanded = torch.cat(edge_index_expanded, dim=1).cuda()
#         output = self.gcn(reshaped_data, edge_index_expanded)
#         x = output.reshape(x.shape[0], 49, 512, num_nodes).permute(0, 3, 1, 2)
#
#         # 49维度
#         edge_index_49 = fully_connected_edge_index(49)
#         reshaped_data = x.permute(0, 1, 3, 2).reshape(-1, 1)  # [batchsize * 49 * 2048 * 13, 1]
#         # 扩展邻接矩阵到所有图上
#         num_graphs = x.shape[0] * 8 * 512  # 图的数量
#         num_nodes = 49  # 每个图的节点数
#         edge_index_expanded = []
#         for i in range(num_graphs):
#             offset = i * num_nodes
#             edge_index_expanded.append(edge_index_49 + offset)
#         edge_index_expanded = torch.cat(edge_index_expanded, dim=1).cuda()
#         output = self.gcn(reshaped_data, edge_index_expanded)
#         x = output.reshape(x.shape[0], 8, 512, num_nodes).permute(0, 1, 3, 2)
#
#         # 512
#         edge_index_512 = fully_connected_edge_index(512)
#         reshaped_data = x.reshape(-1, 1)
#         num_graphs = x.shape[0] * 8 * 49  # 图的数量
#         edge_index_expanded = []
#         for i in range(num_graphs):
#             offset = i * num_nodes
#             edge_index_expanded.append(edge_index_512 + offset)
#         edge_index_expanded = torch.cat(edge_index_expanded, dim=1).cuda()
#         output = self.gcn(reshaped_data, edge_index_expanded)
#         x = output.reshape(x.shape[0], 8, 49, 512)
#         return x
#
#
# def fully_connected_edge_index(num_nodes):
#     """
#     生成全连接图的邻接矩阵 edge_index。
#     :param num_nodes: 节点数量。
#     :return: edge_index, 形状为 [2, num_nodes * (num_nodes - 1)]。
#     """
#     edge_index = []
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if i != j:  # 排除自环
#                 edge_index.append([i, j])
#     edge_index = torch.tensor(edge_index, dtype=torch.int).t().contiguous()
#     return edge_index


