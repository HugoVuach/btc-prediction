from abc import ABC, abstractmethod

class Indicator(ABC):
    def __init__(self, df, **kwargs):
        self.df = df
        self.params = kwargs

    @abstractmethod
    def compute(self):
        """Ajoute les colonnes à self.df"""
        pass