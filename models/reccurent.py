import torch as th
from torch import nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx, hidden_dim, num_classes) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, seq):
        word_embeds = self.word_embeddings(seq) # shape (L, Hin)
        h_0 = self.init_hidden()

        # rnn_out shape (L, Hout), hidden shape (1, Hout)
        rnn_out, hidden = self.rnn(word_embeds, h_0)

        # use \hat{y}_T to make predictions
        inp = rnn_out[-1].view(1, -1)
        out = self.fc(inp)

        return self.softmax(out)


    def init_hidden(self):
        return th.zeros(1, self.hidden_dim, device=th.device('cuda' if th.cuda.is_available() else 'cpu'))
