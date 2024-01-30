import os
from typing import Union
import numpy as np
import toml

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA, FastICA, KernelPCA, NMF, LatentDirichletAllocation

from crossai import generate_transformers
from ._utils import apply_transform

class Transformations:
    
    def __init__(
            self, 
            axis: int,
            val_set: bool = True,
        ) -> None:

        self._axis = axis
        self.coefs = None
        self.splitted = None
        
    
    def transformers(self, config):
        return generate_transformers(self, config)

    def split(
        self,
        xy: tuple,
        test_size: float = None, 
        val_size: float = None,
        random_state: int = None, 
        shuffle: bool = True, 
        stratify: bool = None
    ) -> tuple:
        """Split dataset to train, validation and test splits.

        Args:
            XY (tuple): X, Y data.
            test_size (float, optional): Size of test split. Value range [0.0,1.0]. Defaults to None.
            val_size (float, optional): Size of validation split. Value range [0.0,1.0]. Defaults to None.
            random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to None.
            shuffle (bool, optional): Whether or not to shuffle the data before splitting. Defaults to True.
            stratify (bool, optional): Data is split in a stratified fashion, using this as the class labels. Defaults to None.

        Returns:
            _type_: _description_
        """

        X = xy[0]
        y = xy[1]
        __stratify = None
        data = None
        y = np.asarray(y)
        if len(y.shape) == 1:
            y = y.reshape(-1,1)

        if stratify:
            __stratify = y
        if val_size is None:
            data = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=__stratify
            )
            self.splitted = True
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=__stratify
            )

            if stratify:
                __stratify = y_train

            x_train, x_val, y_train, y_val = train_test_split(
                x_train,
                y_train,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=__stratify
            )

            self.splitted = True
            data = (x_train, x_val, x_test, y_train, y_val, y_test)
        return data
        

    def extra_data(self, xy):
        return (None, xy[0], None, xy[1])

    def pca(
        self,
        data,
        *,
        n_components: int = None, 
        whiten: bool = False,
        svd_solver: str = 'auto', 
        tol: float = 0.0, 
        iterated_power: str = 'auto', 
        n_oversamples: int = 10, 
        power_iteration_normalizer: str = 'auto', 
        random_state: int = None
    ) -> tuple:

        __pca = PCA(
                n_components=n_components,
                whiten=whiten,
                svd_solver=svd_solver,
                tol=tol,
                iterated_power=iterated_power,
                n_oversamples=n_oversamples,
                power_iteration_normalizer=power_iteration_normalizer,
                random_state=random_state
            )

        __data = apply_transform(self, data, func=__pca)
        data = list(data)
        if len(__data) == 3:
            data[0] = __data[0]
            data[1] = __data[1]
            data[2] = __data[2]

        elif len(__data) == 2:
            data[0] = __data[0]
            data[1] = __data[1]

        elif len(__data) == 1:
            data[0] = __data[0]
        
        return tuple(data)

    def nmf(
        self,
        data,
        *,
        n_components=None,
        init=None,
        solver='cd',
        beta_loss='frobenius',
        tol=0.0001,
        max_iter=200,
        random_state=None,
        alpha_W=0.0,
        alpha_H='same',
        l1_ratio=0.0,
        verbose=0,
        shuffle=False,
    ) -> tuple:

        __nmf = NMF(
           n_components=n_components,
           init=init,
           solver=solver,
           beta_loss=beta_loss,
           tol=tol,
           max_iter=max_iter,
           random_state=random_state,
           alpha_W=alpha_W,
           alpha_H=alpha_H,
           l1_ratio=l1_ratio,
           verbose=verbose,
           shuffle=shuffle
       )

        __data = apply_transform(self, data, func=__nmf)
        if len(__data) == 3:
            data[0] = __data[0]
            data[1] = __data[1]
            data[2] = __data[2]

        elif len(__data) == 2:
            data[0] = __data[0]
            data[1] = __data[1]

        elif len(__data) == 1:
            data[0] = __data[0]
        
        return data
    
    def lda(
        self,
        data,
        *,
        n_components:int =10,
        learning_method: str ='batch', 
        learning_decay: float = 0.7, 
        learning_offset:float = 10.0, 
        max_iter: int = 10, 
        batch_size: int = 128, 
        total_samples: float = 1000000.0, 
        perp_tol: float = 0.1, 
        mean_change_tol: float = 0.001, 
        n_jobs: int = None, 
        verbose: bool = 0, 
        random_state: int = None
    ) -> tuple:

        __lda = LatentDirichletAllocation(
        n_components = n_components,
        learning_method = learning_method,
        learning_decay = learning_decay,
        learning_offset = learning_offset,
        max_iter = max_iter,
        batch_size = batch_size,
        total_samples = total_samples,
        perp_tol = perp_tol,
        mean_change_tol = mean_change_tol,
        n_jobs = n_jobs,
        verbose = verbose,
        random_state = random_state,
        )

        __data = apply_transform(self, data, func=__lda)
        data = list(data)
        if len(__data) == 3:
            data[0] = __data[0]
            data[1] = __data[1]
            data[2] = __data[2]

        elif len(__data) == 2:
            data[0] = __data[0]
            data[1] = __data[1]

        elif len(__data) == 1:
            data[0] = __data[0]
        
        return tuple(data)
    
    def shuffle(
        self, 
        data: tuple,
        *,
        random_state: int,
    ) -> tuple:
        """Shuffle data.

        Args:
            data (tuple): Data arrays.
            random_state (int): Determines random number generation for shuffling the data.

        Returns:
            tuple: Shuffled data.
        """

        data = list(data)
        x = data[0]
        y = data[1]

        x, y = shuffle(x, y, random_state=random_state) 

        data[0] = x
        data[1] = y

        return tuple(data)
