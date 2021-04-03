import os
from tqdm import tqdm
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt


class DataLoader(object):
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        if mode == 'train':
            self.path = config.train_path
            self.features_path = os.path.join(self.config.features_path, 'train')
        else:
            self.path = config.validation_path
            self.features_path = os.path.join(self.config.features_path, 'validation')
        self.X = []
        self.y = []
        self.video_ids = []
        self.dataloaders = []
        self.fig = []
        self.ax = []
        for i in range(len(config.plot_samples)):
            fig, ax = plt.subplots(self.config.epochs//self.config.plot_freq+1, 1, figsize=(20, 2*self.config.epochs//self.config.plot_freq+1))
            self.fig.append(fig)
            self.ax.append(ax)
        self.visual_idx = []
        self.load_data()

    def load_data(self):
        action_dict = pickle.load(open(self.config.action_dict_path, 'rb'))
        features = open(self.path, 'r').read().split('\n')[:-1]
        print("Reading data...")
        if self.config.random_sampling:
            np.random.shuffle(features)
            features = features[:len(features)//self.config.random_sampling_ratio]
        if self.mode != 'train':
            features = self.config.plot_samples

        for f in tqdm(range(len(features))):
            feature = features[f]
            if self.config.downsample:
                X = np.load(os.path.join(self.features_path, feature+'.npy'))[:-1][::10,:]
                step = 10
            else: 
                X = np.load(os.path.join(self.features_path, feature+'.npy'))[:-1]
                step = 1
            if feature in self.config.plot_samples:
                self.visual_idx.append(f)
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


