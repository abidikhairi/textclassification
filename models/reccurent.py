import torch as th
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx, hidden_dim, num_classes) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.reccurent_layer = None
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.fc = nn.Linear(hidden_dim, num_classes, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, seq):
        
        if self.reccurent_layer is None:
            raise NotImplementedError('Recurrent layer is not initialized')

        word_embeds = self.word_embeddings(seq) # shape (L, Hin)
        h_0 = self.init_hidden()

        # out shape (L, Hout), hidden shape (1, Hout)
        out, _ = self.reccurent_layer(word_embeds, h_0)
        # use \hat{y}_T to make predictions
        inp = out[-1].view(1, -1)
        out = self.fc(inp)

        return self.softmax(out)


    def init_hidden(self):
        return th.zeros(1, self.hidden_dim, device=th.device('cuda' if th.cuda.is_available() else 'cpu'))

class RNNModel(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        embedding_dim = kwargs['embedding_dim']
        hidden_dim = kwargs['hidden_dim']

        self.reccurent_layer = nn.RNN(embedding_dim, hidden_dim)

    def forward(self, seq):
        return super().forward(seq)

class LSTMModel(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.embedding_dim = kwargs['embedding_dim']
        self.hidden_dim = kwargs['hidden_dim']

        self.reccurent_layer = nn.LSTM(self.embedding_dim, self.hidden_dim)

    def forward(self, seq):
        word_embeds = self.word_embeddings(seq) # shape (L, Hin)
        hidden = self.init_hidden()
        
        # out shape (L, Hout), hidden shape ((1, Hout), (1, Hout))
        out, _ = self.reccurent_layer(word_embeds, hidden)
        # use \hat{y}_T to make predictions
        inp = out[-1].view(1, -1)
        out = self.fc(inp)

        return self.softmax(out)
    
    def init_hidden(self):
        h_0 = th.zeros(1, self.hidden_dim, device=th.device('cuda' if th.cuda.is_available() else 'cpu'))
        c_0 = th.zeros(1, self.hidden_dim, device=th.device('cuda' if th.cuda.is_available() else 'cpu'))

        return (h_0, c_0)


class GRUModel(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        embedding_dim = kwargs['embedding_dim']
        hidden_dim = kwargs['hidden_dim']

        self.reccurent_layer = nn.GRU(embedding_dim, hidden_dim)

    def forward(self, seq):
        return super().forward(seq)