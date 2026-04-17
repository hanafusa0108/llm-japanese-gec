# models/base_model.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def train(self, train_data, dev_data):
        pass

    @abstractmethod
    def predict(self, text):
        pass
