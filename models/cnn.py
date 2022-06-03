import torch as th
from torch import nn


class CnnLstmModel(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=16, padding_idx=0, **kwargs) -> None:
        super().__init__() 

        self.hidden_size = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.conv = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTMCell(input_size=embedding_dim*8, hidden_size=hidden_dim)

        self.sent2label = nn.Linear(in_features=hidden_dim, out_features=num_classes)
        
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=0)


    def init_hidden(self, seq_len: int) -> th.Tensor:
        return th.zeros(seq_len, self.hidden_size).to('cuda:0'), th.zeros(seq_len, self.hidden_size).to('cuda:0')


    def forward(self, seq: th.Tensor) -> th.Tensor:
        embs = self.embedding(seq)
        x = self.conv(embs.unsqueeze(1))
        h_0, c_0 = self.init_hidden(x.size(0))
        
        lstm_in = x.view(x.size(0), -1) # feature concatenation
        lstm_out, _ = self.lstm(lstm_in, (h_0, c_0))
        
        x = self.sent2label(self.dropout(lstm_out[-1]))
        scores = self.softmax(x).unsqueeze(0)

        return scores
