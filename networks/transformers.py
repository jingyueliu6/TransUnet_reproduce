import torch
import torch.nn as nn


class Multi_Head_Attention(nn.Module):
    def __init__(self, d, heads_num):
        '''
        :param d: embedding size(D) in the paper (NxD)
        :param heads_num: multi heads number
        '''
        super(Multi_Head_Attention, self).__init__()
        self.heads = heads_num
        self.input_size = d
        self.to_queries = nn.Linear(d, d*self.heads, bias=False)
        self.to_keys = nn.Linear(d, d*self.heads, bias=False)
        self.to_values = nn.Linear(d, d*self.heads, bias=False)
        # This unifies the outputs of the different heads into
        # a single k-vector
        self.unifyheads = nn.Linear(d*self.heads, d)

    def forward(self, x):
        b, n, d = x.size()
        h = self.heads

        qs = self.to_queries(x).view(b, n, h, d)  # ((b,n,d) -> (b, n, h, d))
        ks = self.to_keys(x).view(b, n, h, d)
        vs = self.to_values(x).view(b, n, h, d)

        qs = qs.permute(0, 2, 1, 3).reshape(b * h, n, d)
        ks = ks.permute(0, 2, 1, 3).reshape(b * h, n, d)
        vs = vs.permute(0, 2, 1, 3).reshape(b * h, n, d)

        w_prime = torch.bmm(qs, ks.permute(0, 2, 1))/(d**0.5)
        w = torch.softmax(w_prime, dim=2)
        y = torch.bmm(w, vs).reshape(b, h, n, d).permute(0, 2, 1, 3).reshape(b, n, -1)
        y = self.unifyheads(y)

        return y


class MLP(nn.Module):
    def __init__(self, d):
        super(MLP, self).__init__()
        self.d = d
        self.fc1 = nn.Linear(d, 4*d)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4*d, d)
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Transformer(nn.Module):
    def __init__(self, d, heads):
        super(Transformer, self).__init__()
        self.multi_heads_att = Multi_Head_Attention(d, heads)
        self.mlp = MLP(d)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)

    def forward(self, x):
        h = x
        x = self.layer_norm1(x)
        x = self.multi_heads_att(x)
        x = x + h
        h = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x+h
        return x




