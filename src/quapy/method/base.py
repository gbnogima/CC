from abc import ABCMeta, abstractmethod

from quapy.dataset.text import LabelledCollection

# Base Quantifier abstract class
# ------------------------------------
class BaseQuantifier(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, data: LabelledCollection, *args): ...

    @abstractmethod
    def quantify(self, documents, *args): ...

    @abstractmethod
    def set_params(self, **parameters): ...

    @abstractmethod
    def get_params(self): ...


