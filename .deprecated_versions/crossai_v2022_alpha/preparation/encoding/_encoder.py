import numpy as np
from sklearn.preprocessing import  LabelEncoder, OneHotEncoder

from crossai import generate_transformers
from ._utils import encode

class Encoder:

    def __init__(self) -> None:

        self.encoder = None
    
    def transformers(self, config):
        return generate_transformers(self, config)

    def one_hot_encoder(self, data):

        """Encode categorical features as a one-hot numeric array.
        """
        return encode(self, data, func=OneHotEncoder(sparse=False))

    def label_encoder(self, data):

        """Encode target labels with value between 0 and n_classes-1.
        """

        return encode(self, data, func=LabelEncoder())
        
    
    def get_encoder(self, _=None):
        return self.encoder