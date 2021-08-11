import logging
from typing import Dict

from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.utils.data as data

def full_label_data(df, tasks):
    selected_rows = np.array([True]*len(df))
    for task in tasks:
        selected_rows = selected_rows & df[task].notnull().to_numpy()
    return selected_rows

class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, 
                args, 
                data_dir, 
                split, 
                full_label_instances = False,
                private_label = "age",
                upsampling = False,
                embedding_type = "text_hidden",
                size = None,
                subsampling = False,
                subsampling_ratio = 1
                ):
        self.args = args
        self.data_dir = Path(data_dir)
        self.dataset_type = {"train", "valid", "test"}
        # check split
        assert split in self.dataset_type, "split should be one of train, valid, and test"
        self.split = split
        
        self.embedding_type = embedding_type

        # Load preprocessed tweets, labels, and tweet ids.
        print("Loading preprocessed Encoded data")
        if not full_label_instances:
            df, text_embedding, label, author_private_label_columns, total_n, selected_n= self.load_dataset(not_nan_col = [],
                                                                                                            author_private_labels = [private_label])
        else:
            df, text_embedding, label, author_private_label_columns, total_n, selected_n = self.load_dataset(not_nan_col = [private_label],
                                                                                                            author_private_labels = [private_label])

        self.X = text_embedding
        self.y = label
        self.private_label = author_private_label_columns[private_label]

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.private_label = np.array(self.private_label)

        # Upsampling
        if full_label_instances and upsampling:
            if size == None:
                # Keep using the same size by default
                pass
            else:
                # Use the target size
                total_n = size
            
            # upsampling with replacement to get a same size dataset
            selected_indices = np.random.choice([i for i in range(selected_n)], replace=True, size=total_n)

            # update values
            self.X = self.X[selected_indices]
            self.y = self.y[selected_indices]
            self.private_label = self.private_label[selected_indices]
        elif full_label_instances and subsampling:
            # keep the same total number
            # reduce the number of distinct instances with private labels
            # 0 <= subsampling_ratio <= 1
            sub_indices = np.random.choice([i for i in range(selected_n)], replace=False, size = int(subsampling_ratio*selected_n))
            print("Number of distinct instances: {}".format(len(sub_indices)))
            
            selected_indices = np.random.choice(sub_indices, replace=True, size=selected_n)
            
            # update values
            self.X = self.X[selected_indices]
            self.y = self.y[selected_indices]
            self.private_label = self.private_label[selected_indices]
        else:
            pass

        print("Done, loaded data shapes: {}, {}, {}".format(self.X.shape, self.y.shape, self.private_label.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index], self.private_label[index]

    def load_dataset(
        self,
        not_nan_col = [],
        author_private_labels = ["age"]
        ):
        dataset_dir = self.data_dir / "{}_set.pkl".format(self.split)
        df = pd.read_pickle(dataset_dir)
        # df = df.replace("x", np.nan)

        total_n = len(df)

        df = df[full_label_data(df, not_nan_col)]

        selected_n = len(df)

        print("Select {} out of {} in total.".format(selected_n, total_n))

        # input and labels
        text_embedding = list(df[self.embedding_type])
        label = list(df["label"])
        # private_labels
        author_private_label_columns = {
            p_name:list(df[p_name].astype("float"))
            for p_name in author_private_labels
        }

        for p_name in author_private_labels:
            df[p_name] = author_private_label_columns[p_name]

        return df, text_embedding, label, author_private_label_columns, total_n, selected_n


if __name__ == "__main__":
    class Args:
        gender_balanced = False
    
    data_path = "path"

    split = "train"
    args = Args()
    my_dataset = HateSpeechDataset(args, 
                                data_path, 
                                split, 
                                full_label_instances = True, 
                                upsampling = False,
                                private_label = "age",
                                embedding_type = "deepmoji_encoding",
                                size=None,
                                subsampling = True,
                                subsampling_ratio = 0.5
                                )