from abc import ABC, abstractmethod

class BaseRunner(ABC):
    def __init__(self,):
        pass

    def get_action(self, data:tuple):
        raise NotImplementedError
    
    def train(self,):
        raise NotImplementedError