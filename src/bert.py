params = None

import ktrain
import numpy as np

class Bert():     

    def __init__(self):
        self.classifier = params['classifier']
        self.model = params['model']
        self.preproc = params['preproc']
        self.cache = {}
        self.parameters = {'confidence': 0.5}

    def fit(self, documents, labels):
        self.classifier.fit_onecycle(lr = 2e-5, epochs = 3)
        self.predictor = ktrain.get_predictor(self.classifier.model, self.preproc)
        return self
    
    def predict(self, documents):
        confidence = self.parameters['confidence']
        hash_docs = hash(str(documents))
        hash_docs = str(hash_docs)

        probs = []
        if hash_docs in self.cache:
          probs = self.cache[hash_docs]
        else:
          probs = self.predictor.predict_proba(documents)
          self.cache[hash_docs] = probs
        
        L = []
        for r in probs:
          if r[0] >= confidence:
            L.append(0)
          else: L.append(1)
        return np.asarray(L)


    def get_params(self):
        return self.parameters
    
    def set_params(self, **parameters):
        self.parameters = parameters
        print('parameters = ',parameters)
        pass