import torch
from math import sqrt
from torch import nn


class RNNCell(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.hidden_size = out_features
        
        self.wx = nn.Parameter(torch.randn(in_features, out_features))
        self.wh = nn.Parameter(torch.randn(out_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.init_prarameters()
    
    def forward(self, xt, ht_prev):
        ht = torch.tanh(torch.matmul(xt, self.wx) + torch.matmul(ht_prev, self.wh) + self.bias)
        return ht

    def init_prarameters(self):
        nn.init.uniform_(self.wx.data, a=-torch.sqrt(1.0/self.hidden_size), b=torch.sqrt(1.0/self.hidden_size))
        nn.init.uniform_(self.wh.data, a=-torch.sqrt(1.0/self.hidden_size), b=torch.sqrt(1.0/self.hidden_size))


class LSTMCell(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.hidden_size = out_features

        self.wf =  nn.Parameter(torch.Tensor(in_features, out_features))
        self.wi =  nn.Parameter(torch.Tensor(in_features, out_features))
        self.wo =  nn.Parameter(torch.Tensor(in_features, out_features))
        self.wc =  nn.Parameter(torch.Tensor(in_features, out_features))

        self.bf =  nn.Parameter(torch.zeros(out_features))
        self.bi =  nn.Parameter(torch.zeros(out_features))
        self.bo =  nn.Parameter(torch.zeros(out_features))
        self.bc =  nn.Parameter(torch.zeros(out_features))

        self.uf =  nn.Parameter(torch.Tensor(out_features))
        self.ui =  nn.Parameter(torch.Tensor(out_features))
        self.uo =  nn.Parameter(torch.Tensor(out_features))
        self.uc =  nn.Parameter(torch.Tensor(out_features))

        self.init_parameters()

    def init_parameters(self):
        self.wf.data.uniform_(-1.0/sqrt(self.hidden_size), 1.0/sqrt(self.hidden_size))
        self.wi.data.uniform_(-1.0/sqrt(self.hidden_size), 1.0/sqrt(self.hidden_size))
        self.wo.data.uniform_(-1.0/sqrt(self.hidden_size), 1.0/sqrt(self.hidden_size))
        self.wc.data.uniform_(-1.0/sqrt(self.hidden_size), 1.0/sqrt(self.hidden_size))

        self.uf.data.uniform_(-1.0/sqrt(self.hidden_size), 1.0/sqrt(self.hidden_size))
        self.ui.data.uniform_(-1.0/sqrt(self.hidden_size), 1.0/sqrt(self.hidden_size))
        self.uo.data.uniform_(-1.0/sqrt(self.hidden_size), 1.0/sqrt(self.hidden_size))
        self.uc.data.uniform_(-1.0/sqrt(self.hidden_size), 1.0/sqrt(self.hidden_size))


    def forward(self, xt, ht_prev, ct_prev):
        ft = torch.sigmoid(torch.matmul(xt, self.wf) + torch.matmul(ht_prev, self.uf) + self.bf)
        it = torch.sigmoid(torch.matmul(xt, self.wi) + torch.matmul(ht_prev, self.ui) + self.bi)
        ot = torch.sigmoid(torch.matmul(xt, self.wo) + torch.matmul(ht_prev, self.uo) + self.bo)
        
        c_prime = torch.tanh(torch.matmul(xt, self.wc) + torch.matmul(ht_prev, self.uc) + self.bc)

        ct = ft * ct_prev + it * c_prime

        ht = ot * torch.tanh(ct)

        return ht, ct


class RNNLayer(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.rnn = RNNCell(in_features, out_features)

    def forward(self, x, h0):
        h = []
        ht = h0

        for xt in x:
            ht = self.rnn(xt, ht)
            h.append(ht)
            
        return torch.vstack(h), ht

class RNNNativeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx, hidden_dim, num_classes) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.rnn = RNNLayer(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes, bias=False)


    def forward(self, seq):
        word_embeds = self.word_embeddings(seq)
        h_0 = self.init_hidden()
        
        rnn_out, hidden = self.rnn(word_embeds, h_0)

        inp = rnn_out[-1].view(1, -1)
        out = self.fc(inp)
        return out

    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim)


class LSTMLayer(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.lstm = LSTMCell(in_features, out_features)

    def forward(self, x, h0, c0):
        h = []
        ht = h0
        ct = c0

        for xt in x:
            ht, ct = self.lstm(xt, ht, ct)
            h.append(ht)
            
        return torch.stack(h), (ht, ct)


class LSTMNativeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx, hidden_dim, num_classes) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = LSTMLayer(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes, bias=False)


    def forward(self, seq):
        word_embeds = self.word_embeddings(seq)
        h_0, c_0 = self.init_hidden()
        
        lstm_out, (hidden, cell) = self.lstm(word_embeds, h_0, c_0)

        inp = lstm_out[-1].view(1, -1)
        out = self.fc(inp)
        return out

    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim)
