import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from quapy.dataset.text import LabelledCollection
from quapy.method.neural import EarlyStop


#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def _get_net_class_from_name(name:str):
    if name == 'lstm':
        net_class = LSTMnet
    elif name == 'cnn':
        net_class = CNNnet
    else:
        raise ValueError(f'unknown network type {name}')
    return net_class


class NeuralClassifierTrainer:

    def __init__(self,
                 net,
                 vocabulary_size,
                 embedding_size=100,
                 hidden_size=512,
                 lr=1e-3,
                 drop_p=0.5,
                 weight_decay=0,
                 device='cpu',
                 checkpointpath='../checkpoint/classifier_net.dat'):

        super().__init__()

        assert net in {'lstm', 'cnn'}, f'unknown net type {net}'
        self.net_type = net
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.to_device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.drop_p = drop_p
        self.checkpointpath = checkpointpath

        self.patience = 20
        self.epochs = 500
        self.batch_size = 100
        self.batch_size_test = 500
        self.classes_ = np.asarray([0, 1])
        self.pad_index = vocabulary_size
        self.padding_length = 300

        print(f'[NeuralNetwork running on {device}]')
        os.makedirs(Path(checkpointpath).parent, exist_ok=True)

    def init_classifier(self):
        net_class = _get_net_class_from_name(self.net_type)
        # +1 is for leaving room for the pad index
        self.net = net_class(self.vocabulary_size + 1,
                             embedding_size=self.embedding_size,
                             hidden_size=self.hidden_size,
                             drop_p=self.drop_p).to(self.to_device)
        self.net.xavier_uniform()

    def get_params(self):
        return {
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'lr': self.lr,
            'drop_p': self.drop_p,
            'weight_decay': self.weight_decay
        }

    def set_params(self, **params):
        self.embedding_size = params.get('embedding_size', self.embedding_size)
        self.hidden_size = params.get('hidden_size', self.hidden_size)
        self.lr = params.get('lr', self.lr)
        self.drop_p = params.get('drop_p', self.drop_p)
        self.weight_decay = params.get('weight_decay', self.weight_decay)

    @property
    def device(self):
        return next(self.net.parameters()).device

    def __update_progress_bar(self, pbar):
        pbar.set_description(f'[{self.net_type.upper()}] training epoch={self.current_epoch} '
                             f'tr-loss={self.status["tr"]["loss"]:.5f} '
                             f'tr-acc={100 * self.status["tr"]["acc"]:.2f}% '
                             f'tr-macroF1={100 * self.status["tr"]["f1"]:.2f}% '
                             f'patience={self.early_stop.patience}/{self.early_stop.PATIENCE_LIMIT} '
                             f'val-loss={self.status["va"]["loss"]:.5f} '
                             f'val-acc={100 * self.status["va"]["acc"]:.2f}% '
                             f'macroF1={100 * self.status["va"]["f1"]:.2f}%')

    def _train_epoch(self, data, status, pbar):
        self.net.train()
        losses, predictions, true_labels = [], [], []
        for xi, yi in data:
            self.optim.zero_grad()
            logits = self.net.forward(xi)
            loss = self.criterion(logits, yi)
            loss.backward()
            self.optim.step()
            losses.append(loss.item())
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            status["loss"] = np.mean(losses)
            predictions.extend((probs > 0.5).tolist())
            true_labels.extend(yi.detach().cpu().numpy().tolist())
            status["acc"] = accuracy_score(true_labels, predictions)
            status["f1"] = f1_score(true_labels, predictions, average='binary', pos_label=self.minoritary_class_)
            self.__update_progress_bar(pbar)

    def _test_epoch(self, data, status, pbar):
        self.net.eval()
        losses, predictions, true_labels = [], [], []
        with torch.no_grad():
            for xi, yi in data:
                logits = self.net.forward(xi)
                loss = self.criterion(logits, yi)
                losses.append(loss.item())
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                predictions.extend((probs > 0.5).tolist())
                true_labels.extend(yi.detach().cpu().numpy().tolist())

            status["loss"] = np.mean(losses)
            status["acc"] = accuracy_score(true_labels, predictions)
            status["f1"] = f1_score(true_labels, predictions, average='binary', pos_label=self.minoritary_class_)
            self.__update_progress_bar(pbar)

    def fit(self, documents, labels):
        self.init_classifier()
        self.minoritary_class_ = 0 if labels.mean() > 0.5 else 1

        train, val = LabelledCollection(documents, labels).split_stratified()
        train_generator = TorchDataset(train.documents, train.labels).asDataloader(
            self.batch_size, shuffle=True, pad_length=self.padding_length, pad_index=self.pad_index, device=self.device)
        valid_generator = TorchDataset(val.documents, val.labels).asDataloader(
            self.batch_size_test, shuffle=False, pad_length=self.padding_length, pad_index=self.pad_index, device=self.device)

        self.status = {'tr': {'loss':-1, 'acc': -1, 'f1':-1},
                       'va': {'loss':-1, 'acc': -1, 'f1':-1}}

        self.criterion = BalancedBCEWithLogitLoss(pos_class_prevalence=labels.mean())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.early_stop = EarlyStop(self.patience, lower_is_better=False)

        with tqdm(range(1, self.epochs + 1)) as pbar:
            for self.current_epoch in pbar:
                self._train_epoch(train_generator, self.status['tr'], pbar)
                self._test_epoch(valid_generator, self.status['va'], pbar)

                self.early_stop(self.status['va']['f1'], self.current_epoch)
                if self.early_stop.IMPROVED:
                    torch.save(self.net.state_dict(), self.checkpointpath)
                elif self.early_stop.STOP:
                    print(f'training ended by patience exhasted; loading best model parameters in {self.checkpointpath} '
                          f'for epoch {self.early_stop.best_epoch}')
                    self.net.load_state_dict(torch.load(self.checkpointpath))
                    break

        print('performing one training pass over the validation set...')
        self._train_epoch(valid_generator, self.status['tr'], pbar)
        print('[done]')

        return self

    def predict(self, documents):
        return self.predict_probability_positive(documents) > 0.5

    def predict_proba(self, documents):
        # returns the probability in the scikit-learn's style (first column for the class 0, second for class 1)
        probs = self.predict_probability_positive(documents)
        return np.vstack([1-probs, probs]).T

    def predict_probability_positive(self, documents):
        self.net.eval()
        with torch.no_grad():
            positive_probs = []
            for xi in TorchDataset(documents).asDataloader(
                    self.batch_size_test, shuffle=False, pad_length=self.padding_length,
                    pad_index=self.pad_index, device=self.device):
                positive_probs.append(self.net.probability(xi))
        return np.concatenate(positive_probs)

    def transform(self, documents):
        self.net.eval()
        embeddings = []
        with torch.no_grad():
            for xi in TorchDataset(documents).asDataloader(
                    self.batch_size_test, shuffle=False, pad_length=self.padding_length,
                    pad_index=self.pad_index, device=self.device):
                embeddings.append(self.net.document_embedding(xi).detach().cpu().numpy())
        return np.concatenate(embeddings)


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, documents, labels=None):
        self.documents = documents
        self.labels = labels

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return {'doc': self.documents[index], 'label': self.labels[index] if self.labels is not None else None}

    def asDataloader(self, batch_size, shuffle, pad_length, pad_index, device):
        def collate(batch):
            data = [torch.LongTensor(item['doc'][:pad_length]) for item in batch]
            data = pad_sequence(data, batch_first=True, padding_value=pad_index).to(device)
            targets = [item['label'] for item in batch]
            if targets[0] is None:
                return data
            else:
                targets = torch.as_tensor(targets, dtype=torch.float32).to(device)
                return [data, targets]

        torchDataset = TorchDataset(self.documents, self.labels)
        return torch.utils.data.DataLoader(torchDataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


class TextClassifierNet(torch.nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def document_embedding(self, x): ...

    def forward(self, x):
        doc_embedded = self.document_embedding(x)
        return self.output(doc_embedded).view(-1)

    def dimensions(self):
        return self.dim

    def probability(self, x):
        logits = self(x)
        return torch.sigmoid(logits).detach().cpu().numpy()

    def xavier_uniform(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                torch.nn.init.xavier_uniform_(p)


class LSTMnet(TextClassifierNet):

    def __init__(self, vocabulary_size, embedding_size, hidden_size, repr_size=100, lstm_nlayers=1, drop_p=0.5):
        super().__init__()
        self.lstm_nlayers = lstm_nlayers
        self.lstm_hidden_size = hidden_size
        self.word_embedding = torch.nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, lstm_nlayers, dropout=drop_p, batch_first=True)
        self.dropout = torch.nn.Dropout(drop_p)

        self.dim = repr_size
        self.doc_embedder = torch.nn.Linear(hidden_size, self.dim)
        self.output = torch.nn.Linear(self.dim, 1)

    def init_hidden(self, set_size):
        var_hidden = torch.zeros(self.lstm_nlayers, set_size, self.lstm_hidden_size)
        var_cell = torch.zeros(self.lstm_nlayers, set_size, self.lstm_hidden_size)
        if next(self.lstm.parameters()).is_cuda:
            var_hidden, var_cell = var_hidden.cuda(), var_cell.cuda()
        return var_hidden, var_cell

    def document_embedding(self, x):
        embedded = self.word_embedding(x)
        rnn_output, rnn_hidden = self.lstm(embedded, self.init_hidden(x.size()[0]))
        abstracted = self.dropout(F.relu(rnn_hidden[0][-1]))
        abstracted = self.doc_embedder(abstracted)
        return abstracted


class CNNnet(TextClassifierNet):

    def __init__(self, vocabulary_size, embedding_size, hidden_size=256, repr_size=100, kernel_heights=[3, 5, 7],
                 stride=1, padding=0, drop_p=0.5):
        super(CNNnet, self).__init__()
        in_channels = 1
        self.word_embedding = torch.nn.Embedding(vocabulary_size, embedding_size)
        self.conv1 = nn.Conv2d(in_channels, hidden_size, (kernel_heights[0], embedding_size), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, hidden_size, (kernel_heights[1], embedding_size), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, hidden_size, (kernel_heights[2], embedding_size), stride, padding)
        self.dropout = nn.Dropout(drop_p)

        self.dim = repr_size
        self.doc_embedder = torch.nn.Linear(len(kernel_heights) * hidden_size, self.dim)
        self.output = nn.Linear(self.dim, 1)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def document_embedding(self, input):
        input = self.word_embedding(input)
        input = input.unsqueeze(1)  # input.size() = (batch_size, 1, num_seq, embedding_length)

        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1)  # all_out.size() = (batch_size, num_kernels*out_channels)
        abstracted = self.dropout(F.relu(all_out))  #  (batch_size, num_kernels*out_channels)
        abstracted = self.doc_embedder(abstracted)
        return abstracted


class BalancedBCEWithLogitLoss(torch.nn.Module):
    def __init__(self, pos_class_prevalence):
        assert pos_class_prevalence!=0 and pos_class_prevalence!=1, \
            'At least one example of both classes is needed to train a binary classifier'
        super().__init__()
        self.pos_weight = 1./pos_class_prevalence
        self.neg_weight = 1./(1-pos_class_prevalence)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, true_labels):
        loss = self.criterion(logits, true_labels)
        weights = torch.full(true_labels.shape, fill_value=self.neg_weight, dtype=torch.float32)
        weights[true_labels == 1] = self.pos_weight
        if loss.is_cuda:
            weights = weights.to('cuda')
        loss = loss * weights
        return loss.mean()


