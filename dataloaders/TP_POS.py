import logging
from typing import Dict

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.utils.data as data

def full_label_data(df, tasks):
    selected_rows = np.array([True]*len(df))
    for task in tasks:
        selected_rows = selected_rows & df[task].notnull().to_numpy()
    return selected_rows

class POS(torch.utils.data.Dataset):
    def __init__(self, 
                data_dir, 
                split, 
                private_label = "age",
                sampling = False,
                target_number = 13463,
                sampling_method = None,
                sampling_threshold = 0,
                silence = False
                ):
        # data file dir
        self.data_dir = Path(data_dir)
        self.split = split
        
        # arguments for TrustPolit SA data sampling
        self.sampling = sampling
        self.target_number = target_number
        self.sampling_method = sampling_method
        self.sampling_threshold = sampling_threshold
        if not silence:
            print("Loading preprocessed Encoded data")
        df, text_input, labels, p_labels = self.load_dataset(author_private_labels = private_label)
        
        self.X = text_input
        self.y = labels
        self.private_label = p_labels
        self.df = df

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.private_label = np.array(self.private_label)
        
        if not silence:
            print("Shape of X: {}".format(self.X.shape))
            print("Shape of y: {}".format(self.y.shape))
            print("Shape of private_label: {}".format(self.private_label.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index], self.private_label[index]

    def load_dataset(
        self,
        author_private_labels = "age"
        ):
        dataset_dir = self.data_dir / "{}.pkl".format(self.split)
        df = pd.read_pickle(dataset_dir)
        
        if self.sampling:
            # sampling based on private attribute predictability of each sent
            predictability_column_name = "{}_correctness_mean".format(author_private_labels)
            p_label_column_name = "{}_label".format(author_private_labels)
            
            if self.sampling_method == None:
                # selected_indices = np.array([i for i in range(len(self.y))])
                pass
            elif self.sampling_method == "random":
                # selected_indices = np.random.choice([i for i in range(len(self.y))], replace=True, size=self.target_number)
                df_group0 = df[df[p_label_column_name]==0]
                df_group1 = df[df[p_label_column_name]==1]
                
                df = pd.concat([df_group0.sample(n= self.target_number//2, replace=True, random_state=1), 
                                df_group1.sample(n= self.target_number - self.target_number//2, replace=True, random_state=1)])
            elif self.sampling_method == "largest_leakage":
                # condidate_indices = [i for i,j in enumerate(list(df[predictability_column_name])) if j >= (1 - self.sampling_threshold)]
                # selected_indices = np.random.choice(condidate_indices, replace=True, size=self.target_number)
                
                df_group0 = df[(df[p_label_column_name]==0) & (df[predictability_column_name]>=(1 - self.sampling_threshold))]
                df_group1 = df[(df[p_label_column_name]==1) & (df[predictability_column_name]>=(1 - self.sampling_threshold))]
                
                df = pd.concat([df_group0.sample(n= self.target_number//2, replace=True, random_state=1), 
                                df_group1.sample(n= self.target_number - self.target_number//2, replace=True, random_state=1)])

            elif self.sampling_method == "smallest_leakage":
                # condidate_indices = [i for i,j in enumerate(list(df[predictability_column_name])) if j <= (0 + self.sampling_threshold)]
                # selected_indices = np.random.choice(condidate_indices, replace=True, size=self.target_number)
                df_group0 = df[(df[p_label_column_name]==0) & (df[predictability_column_name]<=(self.sampling_threshold))]
                df_group1 = df[(df[p_label_column_name]==1) & (df[predictability_column_name]<=(self.sampling_threshold))]
                
                df = pd.concat([df_group0.sample(n= self.target_number//2, replace=True, random_state=1), 
                                df_group1.sample(n= self.target_number - self.target_number//2, replace=True, random_state=1)])
            elif self.sampling_method == "absolute_leakage":
                # condidate_indices = [i for i,j in enumerate(list(df[predictability_column_name])) if (j-0.5)**2 >= (0.5-self.sampling_threshold)**2]
                # selected_indices = np.random.choice(condidate_indices, replace=True, size=self.target_number)
                df_group0_min = df[(df[p_label_column_name]==0) & (df[predictability_column_name]<=(self.sampling_threshold))]
                df_group1_min = df[(df[p_label_column_name]==1) & (df[predictability_column_name]<=(self.sampling_threshold))]
                
                df_group0_max = df[(df[p_label_column_name]==0) & (df[predictability_column_name]>=(1 - self.sampling_threshold))]
                df_group1_max = df[(df[p_label_column_name]==1) & (df[predictability_column_name]>=(1 - self.sampling_threshold))]
                
                df = pd.concat([df_group0_min.sample(n= self.target_number//4, replace=True, random_state=1), 
                                df_group1_min.sample(n= self.target_number//4, replace=True, random_state=1), 
                                df_group0_max.sample(n= self.target_number//4, replace=True, random_state=1), 
                                df_group1_max.sample(n= self.target_number//4, replace=True, random_state=1)]
                                )
            else:
                print("Unknown sampleing method")
                pass

            print("Sampling method: {}".format(self.sampling_method))
        
        total_n = len(df)

        # input and labels
        text_embedding = list(df["text_encoding"])
        label = list(df["POS_label_encoding"])
        # protected labels
        protected_labels = list(df["{}_label".format(author_private_labels)])        
        # print("p_label dist:", Counter(protected_labels))

        return df, text_embedding, label, protected_labels


if __name__ == "__main__":
    

    data_path = "path"

    # split = "we_train"
    # split = "we_valid"
    # split = "we_test"
    split = "TP_SA"
    # split = "TP_POS"
    
    sampling_method_list = [None, "random", "largest_leakage", "smallest_leakage", "absolute_leakage"]
    
    my_dataset = POS(
                    data_path, 
                    split, 
                    private_label = "age",
                    sampling = True,
                    target_number = 13463,
                    sampling_method = sampling_method_list[2],
                    sampling_threshold = 0
                    )
    
    print(my_dataset.private_label)