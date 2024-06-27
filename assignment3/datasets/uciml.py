import os
import pandas as pd

from torch import is_tensor, Tensor, LongTensor
from torch.utils.data import Dataset

class AdultDataset(Dataset):
    """
    Class representing the Adult dataset from UC Irvine ML archive
    """
    def __init__(self, data_dir='data', mode='test', transforms = None):
        """
        Load dataset from preprocessed pickle files
        """
        filename = 'uciml_adult_test.pkl' if mode == 'test' else 'uciml_adult_train.pkl'
        self.data = Tensor(pd.read_pickle(os.path.join(data_dir, filename)).dropna().values)
        self.transforms = transforms

    def get_num_classes(self):
        return 2
    
    def get_num_features(self):
        return 14

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if is_tensor(index):
            index = index.tolist()

        sample = self.data[index, :-1]
        label = self.data[index, -1].long() - 1

        if self.transforms:
            sample = self.transforms(sample)

        return (sample, label)


class DryBeanDataset(Dataset):
    """
    Class representing the Dry Bean dataset from UC Irvine ML archive
    """
    def __init__(self, data_dir='data', mode='test', transforms = None):
        """
        Load dataset from preprocessed pickle files
        """
        filename = 'uciml_drybean.pkl'
        self.data = Tensor(pd.read_pickle(os.path.join(data_dir, filename)).values)
        if transforms:
            temp = transforms(self.data[:, :-1])
            self.data[:, :-1] = temp

    def get_num_classes(self):
        return 7
    
    def get_num_features(self):
        return 16

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if is_tensor(index):
            index = index.tolist()

        sample = self.data[index, :-1]
        label = self.data[index, -1].long()

        return (sample, label)
