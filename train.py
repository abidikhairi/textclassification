import logging
import torch as th
from utils import create_default_parser, load_data, build_vocab


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    dataset = load_data(args.dataset)
    
    trainset = dataset['train']
    testset = dataset['test']
    
    word2idx = build_vocab(trainset, f'data/{args.dataset}-vocab.json')
    import pdb; pdb.set_trace()
    
if __name__ == '__main__':
    parser = create_default_parser()

    main(parser.parse_args())