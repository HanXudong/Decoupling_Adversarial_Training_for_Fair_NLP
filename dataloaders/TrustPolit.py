
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data

class TrustPolitDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_dir, split, random_state = 2020, embedding_type = "avg"):
        self.args = args
        self.data_dir = data_dir
        self.dataset_type = {"train", "dev", "test"}
        # check split
        assert split in self.dataset_type, "split should be one of train, dev, and test."
        # check embedding type
        assert embedding_type in ("avg", "cls"), "Embedding should either be avg or cls."
        self.embedding_type = embedding_type

        self.split = split
        
        self.random_state = random_state
        # self.filename = "merge.en.downsample"
        # self.filename = "processed_20K" # Padding to the left
        # self.filename = "encoded_20K.pkl" # Padding to the right
        # self.filename = "encoded_50K_location_balanced.pkl" 
        # self.filename = "encoded_30K_location.pkl"
        self.filename = "ml_location_balanced.pkl"

        # Init 
        self.X = []
        self.y = []
        # self.gender_label = []
        # self.age_label = []
        self.location_label = []


        # Load preprocessed tweets, labels, and tweet ids.
        print("Loading TP data")
        self.load_data()

        self.X = np.array(self.X)
        self.y = np.array(self.y)      
        # self.gender_label = np.array(self.gender_label)
        # self.age_label = np.array(self.age_label)
        self.location_label = np.array(self.location_label)


        print("Done, loaded data shapes: {}, {}".format(self.X.shape, self.y.shape))




    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index], self.location_label[index]
    
    def load_data(self):
        total_data = np.array(pickle.load(open(self.data_dir+self.filename, "rb") ))
        # print(len(total_data))
        Data_train, Data_test = train_test_split(total_data, test_size=0.1, random_state=self.random_state)
        Data_train, Data_dev = train_test_split(Data_train, test_size=0.11, random_state=self.random_state)

        if self.split == "train":
            Data_partation = Data_train
        elif self.split == "dev":
            Data_partation = Data_dev
        else:
            Data_partation = Data_test
        
        if self.embedding_type == "avg":
            self.X = [list(i) for i in Data_partation[:,0]] #Text avg embeddings
        else:
            self.X = [list(i) for i in Data_partation[:,1]] #Text CLS embeddings
        self.y = Data_partation[:,2].astype(np.float) #Rating
        self.location_label = Data_partation[:,3].astype(np.float) # location
        # self.gender_label = [i[1]+1 for i in Data_partation] # Gender
        # self.age_label = [i[2]+1 for i in Data_partation] # Age

        return 0

if __name__ == "__main__":
    class Args:
        gender_balanced = False
    
    data_path = "path"
    split = "dev"
    args = Args()
    dataset = TrustPolitDataset(args, data_path, split)
    from collections import Counter
    print("target label: {}".format(Counter(dataset.y)))
    # print("age label: {}".format(Counter(dataset.age_label)))
    # print("gender label: {}".format(Counter(dataset.gender_label)))
    print("location label: {}".format(Counter(dataset.location_label )))
    # print(dataset.X[1])