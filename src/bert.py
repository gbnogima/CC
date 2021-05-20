params = None

import ktrain
import numpy as np
import keras
import ktrain
from ktrain import text
from quapy.functional import *


class Bert():     

    def __init__(self):
        self.classifier = params['classifier']
        self.model = params['model']
        self.preproc = params['preproc']
        self.predictor = params['predictor']
        self.cache = {}


    def fit(self, documents, labels):
        # fake fit (model trained!)
        return self
    
    def predict(self, documents):
        confidence = self.parameters['confidence']
        hash_docs = hash(str(documents))
        hash_docs = str(hash_docs)

        probs = []
        if hash_docs in self.cache:
          print('#### predicting (in cache): ',confidence,len(documents))
          probs = self.cache[hash_docs]
        else:
          print('#### predicting: ',confidence,len(documents))
          probs = self.predictor.predict_proba(documents)
          self.cache[hash_docs] = probs
        
        L = []
        for r in probs:
          if r[0] >= confidence:
            L.append(0)
          else: L.append(1)
        #print('#####=== v1: ',np.asarray(L))
        #print('#####=== v2: ',(self.predictor.predict(documents)))
        #return np.asarray(self.predictor.predict(documents))
        return np.asarray(L)


    def get_params(self):
        return self.parameters
        #return None
    
    def set_params(self, **parameters):
        self.parameters = parameters
        print('parameters = ', parameters)
        return