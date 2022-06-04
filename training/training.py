import logging
import torch as th
import torchmetrics.functional as thm


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, optimizer, loss_fn, trainset, text2sequence, word2idx, padding_idx, batch_size, epoch, epochs, device):
    """
    Train the model.

        :param model: The model to train.
        :param optimizer: An optimizer to use.
        :param loss_fn: A loss function to use.
        :param trainset: The training set.
        :param text2sequence: A function to convert text to sequence.
        :param word2idx: A dictionary to convert words to indices.
        :param padding_idx: The index of the padding token.
        :param batch_size: The batch size.
        :param epoch: The current epoch.
        :param epochs: The number of epochs.
        :param device: The device to use.
    """
    model.train()

    batch_labels = []
    batch_preds = []
        
    for idx, row in trainset.iterrows():
        import pdb; pdb.set_trace()

        seq = text2sequence(row['text'], word2idx, padding_idx).to(device)
        label = th.tensor(row['label']).unsqueeze(0).to(device)
            
        scores = model(seq)
        loss = loss_fn(scores, label)

        loss.backward()
            
        batch_labels.append(label.detach().cpu().item())
        batch_preds.append(scores.detach().cpu().tolist())

        if idx > 0 and idx % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

            training_loss = loss_fn(th.tensor(batch_preds).squeeze(1), th.tensor(batch_labels)).item() 

            logger.info(f'Epoch [{epoch}/{epochs}] - Batch [{idx // batch_size}/{len(trainset)//batch_size}] - Loss: {training_loss:.4f}')

    return model


def evaluate(model, loss_fn, testset, text2sequence, word2idx, padding_idx, batch_size, device):
    """
    Evaluate the model.
        :param model: The model to evaluate.
        :param loss_fn: A loss function to use.
        :param testset: The test set.
        :param text2sequence: A function to convert text to sequence.
        :param word2idx: A dictionary to convert words to indices.
        :param padding_idx: The index of the padding token.
        :param batch_size: The batch size.
        :param device: The device to use.
    """
    with th.no_grad():
        model.eval()
        test_labels = []
        test_preds = []
            
        for idx, row in testset.iterrows():
            import pdb; pdb.set_trace()
            seq = text2sequence(row['text'], word2idx, padding_idx).to(device)
            label = th.tensor(row['label']).unsqueeze(0).to(device)
                
            scores = model(seq)
            loss = loss_fn(scores, label)

            test_labels.append(label.detach().cpu().item())
            test_preds.append(scores.detach().cpu().tolist())
            
        test_loss = loss_fn(th.tensor(test_preds).squeeze(1), th.tensor(test_labels)).item()
        test_accuracy = thm.accuracy(th.tensor(test_preds).squeeze(1), th.tensor(test_labels)).item()

    return test_loss, test_accuracy