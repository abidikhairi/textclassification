import pickle
import logging
import torch as th
from torch import nn
from utils import create_default_parser, load_data, build_vocab, text2sequence, init_model
from training.training import train, evaluate


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    dataset, num_classes = load_data(args.dataset)
    
    trainset = dataset['train']
    testset = dataset['test']
    
    word2idx = build_vocab(trainset, f'data/{args.dataset}-vocab.json')
    word2idx['<unk>'] = len(word2idx)
    padding_idx = word2idx['<unk>']

    model_parameters = {
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'vocab_size': len(word2idx),
        'padding_idx': padding_idx,
        'num_classes': num_classes
    }
    model_name, model_class, model = init_model(model=args.model, **model_parameters)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    model.to(device)
    last_accuracy = 0.0

    for epoch in range(args.epochs):
        model = train(model, optimizer, loss_fn, trainset, text2sequence, word2idx, padding_idx, args.batch_size, epoch, args.epochs, device)
            
        test_loss, test_accuracy = evaluate(model, loss_fn, testset, text2sequence, word2idx, padding_idx, args.batch_size, device)

        logger.info(f'Epoch [{epoch+1}/{args.epochs}] - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}')

        if test_accuracy > last_accuracy:
            logger.info(f'Saving model to ./data, Test accuracy {test_accuracy*100:.4f} %')   
            
            with open(f'data/{args.dataset}-{model_name}-hyperparameters.pkl', 'wb') as f:
                pickle.dump((model_class, model_parameters), f)
                
            th.save(model.state_dict(), f'data/{args.dataset}-{model_name}-model.pt')
            last_accuracy = test_accuracy


if __name__ == '__main__':
    parser = create_default_parser()

    main(parser.parse_args())
