import os
import argparse
import torch
import json
from nltk.tokenize import word_tokenize
from datasets import load_dataset


def create_default_parser():
    parser = argparse.ArgumentParser(description='Train text classification model')
    
    parser.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'ag_news', 'yelp_review_full']) # imdb, yelp, ag_news
    parser.add_argument('--model', type=str, default='rnn') # rnn, lstm, gru
    parser.add_argument('-embedding-dim', type=int, default=300)
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5) 
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)

    return parser


def load_data(name, split='train'):
    if name == 'imdb':
        return load_dataset('imdb', 'plain_text', split=split)
    dataset = load_dataset(name, 'default', split)
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
