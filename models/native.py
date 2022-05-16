import torch
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
