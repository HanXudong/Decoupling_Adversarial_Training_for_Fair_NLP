import os
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
from pathlib import Path
import json


class SemEvalDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_dir, split, return_gendered, return_all = False):
        self.args = args
        self.return_gendered = return_gendered
        self.data_dir = data_dir
        self.dataset_type = {"train", "dev", "test"}
        # check split
        assert split in self.dataset_type, "split should be one of train, dev, and test"
        self.split = split
        # Load preprocessed tweets, labels, and tweet ids.
        print("Loading preprocessed RoBERTa format data")
        self.context = torch.load(self.data_dir+"processed_context.pt")[self.split]
        self.tids = torch.load(self.data_dir+"tids.pt")[self.split]
        self.labels = torch.load(self.data_dir+"labels.pt")[self.split]
        print("Done, loaded data shapes: {}, {}, {}".format(self.context.shape, self.labels.shape, self.tids.shape))
        # Load gendered information
        with open(self.data_dir+'GenderInfo.json') as f:
            self.GenderInfo = json.load(f)[self.split]
        self.male_ids = self.GenderInfo["male"]
        self.female_ids = self.GenderInfo["female"]
        # Balancing gender distribution
        if args.gender_balanced:
            number_of_male_instances = number_of_female_instances = min(len(self.male_ids), len(self.female_ids))
        else:
            number_of_female_instances = len(self.female_ids)
            number_of_male_instances = len(self.male_ids)
        self.male_ids = set(self.male_ids[1:number_of_male_instances])
        self.female_ids = set(self.female_ids[1:number_of_female_instances])
        print("Number of Male and Female instances: {} and {}".format(number_of_male_instances, number_of_female_instances))
        
        self.X = []
        self.y = []
        if return_all:
            # Return all data
            self.X = self.context
            self.y = self.labels
        else:
            # Return instances with or withoud gender information
            if return_gendered:
                self.gender_label = []
                for i, j in enumerate(self.tids):
                    if j in self.male_ids:
                        self.gender_label.append(0)
                        self.X.append(self.context[i])
                        self.y.append(self.labels[i])
                    elif j in self.female_ids:
                        self.gender_label.append(1)
                        self.X.append(self.context[i])
                        self.y.append(self.labels[i])
                    else: # gender informatin cannot be retrived
                        pass
            else:
                gendered_ids = self.male_ids.union(self.female_ids)
                for i,j in enumerate(self.tids):
                    if j not in gendered_ids:
                        self.X.append(self.context[i])
                        self.y.append(self.labels[i])
                    else:
                        pass
        print("Number of instances: {}".format(len(self.y)))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.return_gendered:
            return self.X[index], self.y[index], self.gender_label[index]
        else:
            return self.X[index], self.y[index]


if __name__ == "__main__":
    class Args:
        gender_balanced = False
    
    data_path = "path"
    split = "test"
    args = Args()
    args.gender_balanced = False
    _ = SemEvalDataset(args, data_path, split, return_gendered = True)
