import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model

class GAT_GCN_Fingerprint(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GAT_GCN_Fingerprint, self).__init__()

        #fingerprint feature
        self.embedding_xf = nn.Embedding(num_features_xt + 1, embed_dim * 2)

        self.conv_xf_1 = nn.Conv1d(in_channels=167, out_channels=32, kernel_size=8)
        self.pool_xf_1 = nn.MaxPool1d(3)
        self.conv_xf_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8)
        self.pool_xf_2 = nn.MaxPool1d(3)
        self.conv_xf_3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8)
        self.pool_xf_3 = nn.MaxPool1d(3)
        self.fc1_xf = nn.Linear(128 * 6, 1024)
        self.fc2_xf = nn.Linear(1024, output_dim)

        # cell line feature
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim*2)

        self.conv_xt_1 = nn.LSTM(256, embed_dim , num_layers=2, batch_first=True, bidirectional=True)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.LSTM(85, embed_dim , num_layers=2, batch_first=True, bidirectional=True)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.LSTM(85, embed_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.pool_xt_3 = nn.MaxPool1d(3)
        self.fc1_xt = nn.Linear(85 * 735, 1024)
        self.fc2_xt = nn.Linear(1024, output_dim)
        # combined layers
        #self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # protein input feed-forward:
        target = data.target
        #target = target[:,None,:]
        target = target.long()
        target = self.embedding_xt(target)
        #print("=======target.shape=========")
        #print(target.shape)

        # 1d conv layers
        conv_xt = self.conv_xt_1(target)
        conv_xt = F.relu(conv_xt[0])
        conv_xt = self.pool_xt_1(conv_xt)
        # print("==========conv_xt.shape===========")
        # print(conv_xt.shape)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = conv_xt[0]
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = conv_xt[0]
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)
        # print("==========conv_xt.shape===========")
        # print(conv_xt.shape)
        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        #print("=======xt11111111.shape=========")
        #print(xt.shape)
        xt = torch.relu(self.fc1_xt(xt))
        xt = F.dropout(xt, p=0.2, training=self.training)
        #print("=======xt222.shape=========")
        #print(xt.shape)
        xt = self.fc2_xt(xt)
        xt = F.dropout(xt, p=0.2, training=self.training)

        fingerprints = data.fingerprints
        # print("======drug.shape===========")
        # print(drug.shape)
        fingerprints = fingerprints.long()
        embedded_xf = self.embedding_xf(fingerprints)
        conv_xf = self.conv_xf_1(embedded_xf)
        conv_xf = torch.relu(conv_xf)
        conv_xf = self.pool_xf_1(conv_xf)

        conv_xf = self.conv_xf_2(conv_xf)
        conv_xf = torch.relu(conv_xf)
        conv_xf = self.pool_xf_2(conv_xf)

        conv_xf = self.conv_xf_3(conv_xf)
        conv_xf = torch.relu(conv_xf)
        conv_xf = self.pool_xf_3(conv_xf)
        conv_xf = conv_xf.view(-1, conv_xf.shape[1] * conv_xf.shape[2])

        fingerprints = torch.relu(self.fc1_xf(conv_xf))
        fingerprints = F.dropout(fingerprints, p=0.2, training=self.training)
        fingerprints = self.fc2_xf(fingerprints)
        fingerprints = F.dropout(fingerprints, p=0.2, training=self.training)

        # concat
        #xc = torch.cat((x, xt), 1)
        xc = torch.cat((fingerprints,xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = nn.Sigmoid()(out)
        return out, x
