import os
from pathlib import Path

import torch
from torch.nn import MSELoss
from torch.nn.functional import relu
from tqdm import tqdm

from quapy.method.aggregative import *
from settings import SAMPLE_SIZE


class QuaNetTrainer(AggregativeProbabilisticQuantifier):

    def __init__(self, learner, n_epochs=500, tr_iter_per_poch=200, va_iter_per_poch=21, lr=1e-3, patience=10,
                 checkpointpath='../checkpoint/quanet.dat', device='cuda'):
        assert hasattr(learner, 'transform'), \
            f'the learner {learner.__class__.__name__} does not seem to be able to produce document embeddings ' \
                f'since it does not implement the method "transform"'
        assert hasattr(learner, 'predict_proba'), \
            'the learner {learner.__class__.__name__} does not seem to be able to produce posterior probabilities ' \
                f'since it does not implement the method "predict_proba"'
        self.learner = learner
        self.default_parameters = self.learner.get_params()
        self.n_epochs = n_epochs
        self.tr_iter_per_poch = tr_iter_per_poch
        self.va_iter_per_poch = va_iter_per_poch
        self.lr = lr
        self.patience = patience
        self.checkpointpath = checkpointpath
        os.makedirs(Path(checkpointpath).parent, exist_ok=True)
        self.device = device

    def fit(self, data: LabelledCollection, fit_learner=True, *args):
        # split: 40% for training classification, 40% for training quapy, and 20% for validating quapy
        self.learner, unused_data = \
            training_helper(self.learner, data, fit_learner,  ensure_probabilistic=True, train_val_split=0.4)
        train_data, valid_data = unused_data.split_stratified(0.66)  # 0.66 split of 60% makes 40% and 20%

        # compute the posterior probabilities of the documents
        valid_posteriors = self.soft_classify(valid_data.documents)
        train_posteriors = self.soft_classify(train_data.documents)

        # turn documents' indexes into embeddings
        valid_data.documents = self.learner.transform(valid_data.documents)
        train_data.documents = self.learner.transform(train_data.documents)

        # estimate the hard and soft stats tpr and fpr of the classifier
        valid_predictions = (valid_posteriors>=0.5)
        self.hard_tpr, self.hard_fpr = classifier_tpr_fpr_from_predictions(valid_predictions, valid_data.labels)
        self.soft_tpr, self.soft_fpr = classifier_tpr_fpr_from_predictions(valid_posteriors,  valid_data.labels)
        self.tr_prev = data.prevalence()

        self.status = {
            'tr-loss': -1,
            'va-loss': -1,
        }

        self.criterion = MSELoss()
        self.quanet = QuaNetModule(
            train_data.documents.shape[1], lstm_hidden_size=128, lstm_nlayers=2, stats_size=10).to(self.device)
        self.optim = torch.optim.Adam(self.quanet.parameters(), lr=self.lr)
        self.early_stop = EarlyStop(self.patience, lower_is_better=True)
        for self.current_epoch in range(1, self.n_epochs):
            self.epoch(train=True,  data=train_data, posteriors=train_posteriors, iterations=self.tr_iter_per_poch)
            self.epoch(train=False, data=valid_data, posteriors=valid_posteriors, iterations=self.va_iter_per_poch)

            self.early_stop(self.status['va-loss'], self.current_epoch)
            if self.early_stop.IMPROVED:
                torch.save(self.quanet.state_dict(), self.checkpointpath)
            elif self.early_stop.STOP:
                print(f'training ended by patience exhasted; '
                      f'loading best model parameters in {self.checkpointpath} '
                      f'for epoch {self.early_stop.best_epoch}')
                self.quanet.load_state_dict(torch.load(self.checkpointpath))
                self.epoch(train=True, data=valid_data, posteriors=valid_posteriors, iterations=self.va_iter_per_poch)
                break

        return self

    def get_aggregative_estims(self, posteriors):
        cc_prev = prevalence_from_probabilities(posteriors, binarize=True)
        acc_prev = adjusted_quantification(cc_prev, self.hard_tpr, self.hard_fpr)
        pcc_prev = prevalence_from_probabilities(posteriors, binarize=False)
        pacc_prev = adjusted_quantification(pcc_prev, self.soft_tpr, self.soft_fpr)
        emq = ExpectationMaximizationQuantifier.EM(self.tr_prev, posteriors)

        estimations = [cc_prev, acc_prev, pcc_prev, pacc_prev, emq]
        # print(estimations)
        estimations.extend([1-estim for estim in estimations])

        return estimations

    def quantify(self, documents, *args):
        posteriors = self.soft_classify(documents)
        embeddings = self.learner.transform(documents)
        quant_estims = self.get_aggregative_estims(posteriors)
        self.quanet.eval()
        with torch.no_grad():
            prevalence = self.quanet.forward(embeddings, posteriors, quant_estims).item()
        return prevalence

    def epoch(self, train, data: LabelledCollection, posteriors, iterations):
        self.quanet.train(mode=train)
        losses = []
        prevalences = np.random.rand(iterations) if train else prevalence_linspace(iterations,1,0)
        with tqdm(enumerate(prevalences)) as pbar:
            for it, p in pbar:
                index = data.sampling_index(p, SAMPLE_SIZE)
                sample_data = data.sampling_from_index(index)
                sample_posteriors = posteriors[index]
                quant_estims = self.get_aggregative_estims(sample_posteriors)

                if train:
                    self.optim.zero_grad()
                phat = self.quanet.forward(sample_data.documents, sample_posteriors, quant_estims)
                loss = self.criterion(phat, torch.as_tensor([p], dtype=torch.float, device=self.device))
                if train:
                    loss.backward()
                    self.optim.step()
                losses.append(loss.item())

                self.status['tr-loss' if train else 'va-loss'] = np.mean(losses)
                pbar.set_description(f'[QuaNet][{"training" if train else "validating"}] '
                                     f'epoch={self.current_epoch} [it={it}/{iterations}]\t'
                                     f'tr-loss={self.status["tr-loss"]:.5f} '
                                     f'val-loss={self.status["va-loss"]:.5f} '
                                     f'patience={self.early_stop.patience}/{self.early_stop.PATIENCE_LIMIT}')



class QuaNetModule(torch.nn.Module):
    def __init__(self,
                 doc_embedding_size,
                 stats_size,
                 lstm_hidden_size=64,
                 lstm_nlayers=1,
                 ff_layers=[1024, 512],
                 bidirectional=True,
                 drop_p=0.5):
        super().__init__()

        self.hidden_size = lstm_hidden_size
        self.nlayers = lstm_nlayers
        self.bidirectional = bidirectional
        self.ndirections = 2 if self.bidirectional else 1
        self.drop_p = drop_p
        self.lstm = torch.nn.LSTM(doc_embedding_size+2,  # +2 stands for the pos/neg posterior probs. (concatenated)
                                  lstm_hidden_size, lstm_nlayers, bidirectional=bidirectional,
                                  dropout=drop_p, batch_first=True)
        self.dropout = torch.nn.Dropout(self.drop_p)

        lstm_output_size = self.hidden_size * self.ndirections
        ff_input_size = lstm_output_size + stats_size

        prev_size = ff_input_size
        self.ff_layers = torch.nn.ModuleList()
        for lin_size in ff_layers:
            self.ff_layers.append(torch.nn.Linear(prev_size, lin_size))
            prev_size = lin_size
        self.output = torch.nn.Linear(prev_size, 1)

    @property
    def device(self):
        return 'cuda' if next(self.parameters()).is_cuda else 'cpu'

    def init_hidden(self):
        directions = 2 if self.bidirectional else 1
        var_hidden = torch.zeros(self.nlayers * directions, 1, self.hidden_size)
        var_cell = torch.zeros(self.nlayers * directions, 1, self.hidden_size)
        if next(self.lstm.parameters()).is_cuda:
            var_hidden, var_cell = var_hidden.cuda(), var_cell.cuda()
        return var_hidden, var_cell

    def forward(self, doc_embeddings, doc_posteriors, statistics):
        device = self.device
        neg_posteriors = (1.-doc_posteriors)
        doc_embeddings = torch.as_tensor(doc_embeddings, dtype=torch.float, device=device)
        doc_posteriors = torch.as_tensor(doc_posteriors, dtype=torch.float, device=device)
        neg_posteriors = torch.as_tensor(neg_posteriors, dtype=torch.float, device=device)
        statistics = torch.as_tensor(statistics, dtype=torch.float, device=device)

        order = torch.argsort(doc_posteriors)
        doc_embeddings = doc_embeddings[order]
        doc_posteriors = doc_posteriors[order]
        neg_posteriors = neg_posteriors[order]

        embeded_posteriors = torch.cat((doc_embeddings, doc_posteriors.view(-1,1), neg_posteriors.view(-1,1)), dim=-1)

        # the entire set represents only one instance in quapy contexts, and so the batch_size=1
        # the shape should be (1, number-of-documents, embedding-size + 1)
        embeded_posteriors = embeded_posteriors.unsqueeze(0)

        _, (rnn_hidden,_) = self.lstm(embeded_posteriors, self.init_hidden())
        rnn_hidden = rnn_hidden.view(self.nlayers, self.ndirections, -1, self.hidden_size)
        quant_embedding = rnn_hidden[0].view(-1)
        quant_embedding = torch.cat((quant_embedding, statistics))

        abstracted = quant_embedding.unsqueeze(0)
        for linear in self.ff_layers:
            abstracted = self.dropout(relu(linear(abstracted)))

        logits = self.output(abstracted).view(1)
        prevalence = torch.sigmoid(logits)

        prevalence = ((prevalence-0.5)*1.2 + 0.5) # scales the sigmoids so that the net is able to reach either 1 or 0
        if not self.training:
            prevalence = torch.clamp(prevalence, 0, 1)

        return prevalence


class EarlyStop:

    def __init__(self, patience, lower_is_better=True):
        self.PATIENCE_LIMIT = patience
        self.better = lambda a,b: a<b if lower_is_better else a>b
        self.patience = patience
        self.best_score = None
        self.best_epoch = None
        self.STOP = False
        self.IMPROVED = False

    def __call__(self, watch_score, epoch):
        self.IMPROVED = (self.best_score is None or self.better(watch_score, self.best_score))
        if self.IMPROVED:
            self.best_score = watch_score
            self.best_epoch = epoch
            self.patience = self.PATIENCE_LIMIT
        else:
            self.patience -= 1
            if self.patience <= 0:
                self.STOP = True

