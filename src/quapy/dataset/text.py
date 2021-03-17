import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from quapy.functional import prevalence_linspace, natural_prevalence_sampling
from quapy.util import parallelize

import keras
import ktrain
from ktrain import text


class LabelledCollection:

    def __init__(self, documents, labels):
        self.documents = documents if issparse(documents) else np.asarray(documents)
        self.labels = np.asarray(labels, dtype=int)
        if set(self.labels) not in [{0, 1}, {0}, {1}]:
            raise NotImplementedError('Only binary classification is currently supported')
        n_docs = len(self)
        self.negative_indexes = np.arange(n_docs)[self.labels == 0]
        self.positive_indexes = np.arange(n_docs)[self.labels == 1]

    @classmethod
    # File fomart <0 or 1>\t<document>\n
    def from_file(self, path):
        all_sentences, all_labels = [], []
        for line in tqdm(open(path, 'rt').readlines(), f'loading {path}'):
            line = line.strip()
            if line:
                label, sentence = line.split('\t')
                sentence = sentence.strip()
                label = int(label)
                if sentence:
                    all_sentences.append(sentence)
                    all_labels.append(label)
        return LabelledCollection(all_sentences, all_labels)

    def __len__(self):
        return self.documents.shape[0]

    def prevalence(self):
        return self.labels.mean()

    def num_positives(self):
        return len(self.positive_indexes)

    def num_negatives(self):
        return len(self.negative_indexes)

    def sampling_index(self, prevalence, size, shuffle=True):
        assert 0 <= prevalence <= 1, 'prevalence out of range'

        n_pos_requested = int(size * prevalence)
        n_neg_requested = size - n_pos_requested

        n_pos_actual = len(self.positive_indexes)
        n_neg_actual = len(self.negative_indexes)

        pos_indexes_sample = self.positive_indexes[
            np.random.choice(n_pos_actual, size=n_pos_requested, replace=(n_pos_requested > n_pos_actual))
        ] if n_pos_requested > 0 else []

        neg_indexes_sample = self.negative_indexes[
            np.random.choice(n_neg_actual, size=n_neg_requested, replace=(n_neg_requested > n_neg_actual))
        ] if n_neg_requested > 0 else []

        index = np.concatenate([pos_indexes_sample, neg_indexes_sample]).astype(int)

        if shuffle:
            index = np.random.permutation(index)

        return index

    def sampling(self, prevalence, size, shuffle=True):
        index = self.sampling_index(prevalence, size, shuffle)
        return self.sampling_from_index(index)

    def sampling_from_index(self, index):
        documents = self.documents[index]
        labels = self.labels[index]
        return LabelledCollection(documents, labels)

    def split_stratified(self, train_size=0.6):
        tr_docs, te_docs, tr_labels, te_labels = \
            train_test_split(self.documents, self.labels, train_size=train_size, stratify=self.labels)
        return LabelledCollection(tr_docs, tr_labels), LabelledCollection(te_docs, te_labels)

    def artificial_sampling_generator(self, sample_size, n_prevalences=21, repeats=100, smooth_limits=0.0):
        for p in prevalence_linspace(n_prevalences=n_prevalences, repeat=repeats, smooth_limits_epsilon=smooth_limits):
            yield self.sampling(p, sample_size)

    def natural_sampling_generator(self, sample_size, n_samples=2100, prev_std=0.1):
        for p in natural_prevalence_sampling(n_samples, self.prevalence(), prev_std):
            yield self.sampling(p, sample_size)

    def undersampling(self, prevalence):
        current_prevalence = self.prevalence()
        if current_prevalence>prevalence:
            size = self.num_negatives() / (1-prevalence)
        else:
            size = self.num_positives() / prevalence
        return self.sampling(prevalence, size=int(size), shuffle=True)



class TQDataset:

    def __init__(self, training: LabelledCollection, test: LabelledCollection):
        self.training = training
        self.test = test

    def tfidfvectorize(self, min_freq=3):
        vectorizer = TfidfVectorizer(min_df=min_freq, sublinear_tf=True)
        self.training.documents = vectorizer.fit_transform(self.training.documents)
        self.test.documents = vectorizer.transform(self.test.documents)
        self.vocabulary_ = vectorizer.vocabulary_

    def index(self):
        index = Index()
        self.training.documents = index.fit_transform(self.training.documents)
        self.test.documents = index.transform(self.test.documents, n_jobs=-1)
        self.vocabulary_ = index.vocabulary_
    
    def bert_preprocessing(self):
        (x_train, y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=self.training.documents, y_train=self.training.labels,
                                                                        val_pct=0.1,
                                                                        class_names=[0, 1],
                                                                        preprocess_mode='bert',
                                                                        maxlen=64, 
                                                                        max_features=35000)

        model = text.text_classifier(name = 'bert',
                            train_data = (x_train, y_train),
                            preproc = preproc)

        classifier = ktrain.get_learner(model, 
                            train_data=(x_train, y_train), 
                            val_data=(x_test, y_test),
                            batch_size=64
                            )
        self.vocabulary_ = []
        return {'preproc': preproc, 'model': model, 'classifier': classifier}
        
        
    @classmethod
    def from_files(cls, train_path, test_path):
        training = LabelledCollection.from_file(train_path)
        test = LabelledCollection.from_file(test_path)
        return TQDataset(training, test)


class Index:
    def __init__(self, **kwargs):
        """
        :param kwargs: keyworded arguments from _sklearn.feature_extraction.text.CountVectorizer_
        """
        self.vect = CountVectorizer(**kwargs)
        self.unk = -1  # a valid index is assigned after fit

    def fit(self, X):
        """
        :param X: a list of strings
        :return: self
        """
        self.vect.fit(X)
        self.analyzer = self.vect.build_analyzer()
        self.vocabulary_ = self.vect.vocabulary_
        self.unk = self.add_word('UNK')
        return self

    def transform(self, X, n_jobs=-1):
        # given the number of tasks and the number of jobs, generates the slices for the parallel threads
        assert self.unk > 0, 'transform called before fit'
        indexed = parallelize(func=self.index, args=X, n_jobs=n_jobs)
        return np.asarray(indexed)

    def index(self, documents):
        vocab = self.vocabulary_.copy()
        return [[vocab.get(word, self.unk) for word in self.analyzer(doc)] for doc in tqdm(documents, 'indexing')]

    def fit_transform(self, X, n_jobs=-1):
        return self.fit(X).transform(X, n_jobs=n_jobs)

    def vocabulary_size(self):
        return len(self.vocabulary_) + 1  # the reserved unk token

    def add_word(self, word):
        if word in self.vocabulary_:
            raise ValueError(f'word {word} already in dictionary')
        self.vocabulary_[word] = len(self.vocabulary_)
        return self.vocabulary_[word]

