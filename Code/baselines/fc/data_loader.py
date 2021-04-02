import os
from tqdm import tqdm
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt


class DataLoader(object):
    def __init__(self, config, dataset):
        self.config = config
        # self.train_mode = train_mode
        if dataset == 'train':
            self.path = config.train_path
            self.features_path = os.path.join(self.config.features_path, 'train')
        else:
            self.path = config.validation_path
            self.features_path = os.path.join(self.config.features_path, 'validation')
        self.X = []
        self.y = []
        self.video_ids = []
        self.dataloaders = []
        self.visualize_idx = None
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        plt.axis('off')
        plt.xlim([-600, 2000])
        plt.ylim([20-100*(config.epochs//self.config.epochs+5), 100])
        self.load_data()

    def load_data(self):
        action_dict = pickle.load(open(self.config.action_dict_path, 'rb'))
        features = open(self.path, 'r').read().split('\n')[:-1]
        print("Reading data...")
        if self.config.random_sampling:
            np.random.shuffle(features)
            features = features[:len(features)//2]

        for feature in tqdm(features):
            if self.config.downsample:
                X = np.load(os.path.join(self.features_path, feature+'.npy'))[:-1][::10,:]
                step = 10
            else: 
                X = np.load(os.path.join(self.features_path, feature+'.npy'))[:-1]
                step = 1
            y = []
            verbs = open(os.path.join(self.config.gt_path,
                                        feature+'.txt'), 'r').read().split('\n')[:-1]
            for i in range(0, len(verbs), step):
                v = verbs[i]
                if v == 'background':
                    y.append(self.config.num_classes - 1)
                else:
                    y.append(action_dict[v])
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(X), torch.from_numpy(np.array(y)))
            self.video_ids.append(feature)
            self.X.append(X)
            self.y.append(y)
            self.dataloaders.append(
                torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size))
        self.visualize_idx = np.random.randint(len(self.y), size=1)[0]


