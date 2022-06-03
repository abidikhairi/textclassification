import os
import argparse
import torch
import json
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from models.reccurent import RNNModel, LSTMModel, GRUModel
from models.native import RNNNativeModel
from models.cnn import CnnLstmModel


def create_default_parser():
    parser = argparse.ArgumentParser(description='Train text classification model')
    
    parser.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'ag_news', 'yelp_review_full']) # imdb, yelp, ag_news
    parser.add_argument('--model', type=str, default='rnn') # rnn, lstm, gru, rnn-native
    parser.add_argument('--embedding-dim', type=int, default=300)
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5) 
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)

    return parser


def load_data(name, split='train'):
    if name == 'imdb':
        return load_dataset('imdb', 'plain_text', split=split), 2
    elif name == 'ag_news':
        dataset = load_dataset(name, 'default', split)
        return dataset, 4
    return dataset


def build_vocab(dataset, path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            vocab = json.load(f)
            return vocab
    vocab = {}
    for data in dataset:
        tokens = word_tokenize(data['text'])
        for word in tokens:
            if word not in vocab:
                vocab[word] = len(vocab)
    with open(path, 'w') as f:
        json.dump(vocab, f)
    return vocab


def text2sequence(text, word2idx, padding_idx):
    tokens = word_tokenize(text)
    seq = [word2idx[word] if word in word2idx else padding_idx for word in tokens]
    return torch.tensor(seq, dtype=torch.long)

def init_model(**kwargs):
    model_name = kwargs.pop('model')
    model_class = None
    model = None 

    if model == 'rnn':
        model = RNNModel(**kwargs)
        model_class = RNNModel
    elif model_name == 'rnn-native':
        model = RNNNativeModel(**kwargs)
    elif model_name == 'lstm':
        model = LSTMModel(**kwargs)
        model_class = LSTMModel
    elif model_name == 'gru':
        model = GRUModel(**kwargs)
        model_class = GRUModel
    elif model_name == 'cnn':
        model = CnnLstmModel(**kwargs)
        model_class = CnnLstmModel
    else:
        raise ValueError('Unknown model: {}'.format(model_name))
    
    return model_name, model_class, model
