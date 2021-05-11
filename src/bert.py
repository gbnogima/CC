params = None

import ktrain
import numpy as np

class Bert():     

    def __init__(self):
        self.classifier = params['classifier']
        self.model = params['model']
        self.preproc = params['preproc']

    def fit(self, documents, labels):
        self.classifier.fit_onecycle(lr = 2e-5, epochs = 3)
        self.predictor = ktrain.get_predictor(self.classifier.model, self.preproc)
        return self
    
    def predict(self, documents):
        return np.asarray(self.predictor.predict(documents))
    
    def get_params(self):
        return None
    
    def set_params(self, **parameters):
        pass