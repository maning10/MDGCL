import torch as th
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from parms_setting import settings
args = settings()

class DTF(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

class DFAM(nn.Module):
    def __init__(self, in_size, dropout, num_heads=8, hidden_size=128):
        # print(in_size)
        super(DFAM, self).__init__()
        assert in_size % num_heads == 0, "in_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_size // num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.query = nn.Linear(in_size, in_size)
        self.key = nn.Linear(in_size, in_size)
        self.value = nn.Linear(in_size, in_size)

        self.aggregate = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            DTF(hidden_size),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.dropout = nn.Dropout(self.dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, z):
        batch_size, seq_len, _ = z.shape
        # 生成Q, K, V并拆分为多头
        Q = self.query(z).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(z).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(z).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = nn.CELU(alpha=2.0)(output)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        w = self.aggregate(output)
        beta = torch.softmax(w, dim=1)
        aggregated = (beta * output).sum(1)

        return aggregated, beta.squeeze(-1)

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)


class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        ret = self.fc(x)
        return ret
class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)

    def forward(self, h1, h2, h3, h4, c1, c2):
        c_x1 = c1.expand_as(h1).contiguous()
        c_x2 = c2.expand_as(h2).contiguous()

        # positive
        sc_1 = self.fn(h1, c_x1).squeeze(1)
        sc_2 = self.fn(h2, c_x2).squeeze(1)

        # negative
        sc_3 = self.fn(h3, c_x1).squeeze(1)
        sc_4 = self.fn(h4, c_x2).squeeze(1)

        logits = th.cat((sc_1, sc_2, sc_3, sc_4))

        return logits


class DGCT(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(DGCT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels))
        self.dyt_gamma=DTF(num_channels)
        self.dyt_beta=DTF(num_channels)
        self.gamma = nn.Parameter(torch.zeros(1, num_channels))
        self.beta = nn.Parameter(torch.zeros(1, num_channels))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

       
    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum(dim=0, keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            gamma=self.dyt_gamma(embedding)
            norm = gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum(dim=0, keepdim=True) * self.alpha
            gamma=self.dyt_gamma(embedding)
            norm = gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            raise ValueError("Unknown mode!")
        beta=self.dyt_beta(embedding)
        gate = 1. + nn.CELU(alpha=2.0)(embedding * norm + beta)
        return x * gate

class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(Encoder, self).__init__()
        self.gc1a = GCNConv(nfeat, nhid)
        self.gelu1a = nn.CELU(alpha=2.0)
        self.gct1 = DGCT(num_channels=nfeat)  # 修改为输入特征维度
        self.gct2 = DGCT(num_channels=nfeat)   # 修改为隐藏层维度
        self.gc2 = GCNConv(nhid, out)
        self.gelu2 =  nn.CELU(alpha=2.0)
        self.dropout = dropout
        self.norm1 = DTF(nhid)
        self.norm2 = DTF(out)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x, adj):
        # print(x.shape)
        x0 = self.gct1(x)
        x1a = self.gelu1a(self.gc1a(x0, adj))
        x1a = self.norm1(x1a)
        x1a = F.dropout(x1a, self.dropout, training=self.training)
        x1aa = self.norm2(self.gelu2(self.gc2(x1a, adj)))

        x00 = self.gct2(x)
        x1b = self.norm1(self.gc1a(x00, adj))
        x1b = F.dropout(x1b, self.dropout, training=self.training)
        x1b = self.sigmoid(self.gc2(x1b, adj))
        x = x1aa * x1b
        return x


class MDGCL(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, decoder1,drop):
        super(MDGCL, self).__init__()

        self.encoder1 = Encoder(in_dim, hid_dim, out_dim,drop)
        self.encoder2 = Encoder(in_dim, hid_dim, out_dim,drop)

        self.encoder3 = Encoder(in_dim, hid_dim, out_dim,drop)
        self.encoder4 = Encoder(in_dim, hid_dim, out_dim,drop)
        self.dropout = drop
        self.pooling = AvgReadout()
        self.attention = DFAM(out_dim,self.dropout)
        self.disc = Discriminator(out_dim)
        self.act_fn = nn.Sigmoid()
        self.local_mlp = nn.Linear(out_dim, out_dim)
        self.global_mlp = nn.Linear(out_dim, out_dim)
        self.decoder1 = nn.Linear(out_dim * 4, decoder1)
        self.decoder2 = nn.Linear(decoder1, 1)
        self.gelu =  nn.CELU(alpha=2.0)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, data_s, data_f, idx ):
        feat, s_graph = data_s.x, data_s.edge_index
        # print("feat",feat.shape, s_graph.shape)
        shuff_feat, f_graph = data_f.x, data_f.edge_index
        # print("feat", shuff_feat.shape, f_graph.shape)
        h1 = self.encoder1(feat, s_graph)
        h1 = F.dropout(h1, self.dropout, training=self.training)
        h2 = self.encoder2(feat, f_graph)
        h2 = F.dropout(h2, self.dropout, training=self.training)

        h1 = self.gelu(self.local_mlp(h1))
        h2 = self.gelu(self.local_mlp(h2))

        h3 = self.encoder1(shuff_feat, s_graph)
        h3 = F.dropout(h3, self.dropout, training=self.training)
        h4 = self.encoder2(shuff_feat, f_graph)
        h4 = F.dropout(h4, self.dropout, training=self.training)

        h3 = self.gelu(self.local_mlp(h3))
        h4 = self.gelu(self.local_mlp(h4))

        h5 = self.encoder3(feat, s_graph)
        h6 = self.encoder3(feat, f_graph)

        c1 = self.act_fn(self.global_mlp(self.pooling(h1)))
        c2 = self.act_fn(self.global_mlp(self.pooling(h2)))

        out = self.gelu(self.disc(h1, h2, h3, h4, c1, c2))
        h_com = (h5 + h6)/2

        emb = torch.stack([h1, h2, h_com], dim=1)

        # print("emd: ",emb.shape)
        emb, att = self.attention(emb)

        if args.task_type == 'LDA':
            # # dataset1
            entity1 = emb[idx[0]]
            entity2 = emb[idx[1] + 386]
            # dataset2
            # entity1 = emb[idx[0]]
            # entity2 = emb[idx[1] + 230]
        if args.task_type == 'MDA':
            entity1 = emb[idx[0] + 702]
            entity2 = emb[idx[1] + 386]
            # # dataset2
            # entity1 = emb[idx[0] + 635]
            # entity2 = emb[idx[1] + 230]
        if args.task_type == 'LMI':
            entity1 = emb[idx[0]]
            entity2 = emb[idx[1] + 702]
        # multi-relationship modelling decoder
        add = entity1 + entity2
        product = entity1 * entity2
        concatenate = torch.cat((entity1, entity2), dim=1)

        feature = torch.cat((add, product, concatenate), dim=1)

        log1 = F.celu(self.decoder1(feature),alpha=2.0)
        log = self.decoder2(log1)
        return out, log






