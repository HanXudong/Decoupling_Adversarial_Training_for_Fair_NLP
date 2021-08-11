from typing import Dict

from pathlib import Path

import numpy as np

import torch
import torch.utils.data as data

class DeepMojiDataset(torch.utils.data.Dataset):
    def __init__(self, 
                args, 
                data_dir, 
                split, 
                ratio: float = 0.5, 
                n: int = 100000,
                limit_instances_ratio = 0
                ):
        self.args = args
        self.data_dir = Path(data_dir)
        self.dataset_type = {"train", "dev", "test"}
        # check split
        assert split in self.dataset_type, "split should be one of train, dev, and test"
        self.split = split
        self.data_dir = self.data_dir / self.split
        self.ratio = ratio
        self.n = n
        # Init 
        self.X = []
        self.y = []
        self.private_label = []

        # Load preprocessed tweets, labels, and tweet ids.
        print("Loading preprocessed deepMoji Encoded data")
        self.load_data()

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.private_label = np.array(self.private_label)

        assert (limit_instances_ratio>=0) & (limit_instances_ratio<=1), "ratio must be in the range from 0 to 1"
        if (limit_instances_ratio == 0 or limit_instances_ratio == 1):
            pass
        else:
            # subsampling
            number_indices = len(self.y)
            sub_indices = np.random.choice([i for i in range(number_indices)], replace=False, size = int(number_indices*limit_instances_ratio))
            print("Number of distinct instances: {}".format(len(sub_indices)))
            # upsampling with replacement to get a same size dataset
            selected_indices = np.random.choice(sub_indices, replace=True, size=number_indices)

            # update values
            self.X = self.X[selected_indices]
            self.y = self.y[selected_indices]
            self.private_label = self.private_label[selected_indices]

        print("Done, loaded data shapes: {}, {}, {}".format(self.X.shape, self.y.shape, self.private_label.shape))




    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index], self.private_label[index]
    
    def load_data(self):
        # ratios for pos / neg
        n_1 = int(self.n * self.ratio / 2)
        n_2 = int(self.n * (1 - self.ratio) / 2)

        for file, label, private, class_n in zip(['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg'],
                                                [1, 1, 0, 0],
                                                [1, 0, 1, 0],
                                                [n_1, n_2, n_2, n_1]):
            data_dir = self.data_dir / '{}.npy'.format(file)
            data = np.load(data_dir)
            data = list(data[:class_n])
            self.X = self.X + data
            self.y = self.y + [label]*len(data)
            self.private_label = self.private_label + [private]*len(data)

if __name__ == "__main__":
    class Args:
        gender_balanced = False
    
    data_path = "path"
    split = "train"
    args = Args()
    my_dataset = DeepMojiDataset(args, data_path, split)
    for i in range(3):
        print(my_dataset[i])