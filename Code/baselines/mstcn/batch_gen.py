#!/usr/bin/python2.7

import torch
import numpy as np
import random
import copy
import os
import matplotlib.pyplot as plt

class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, rrev_dict, gt_path, features_path, color_path, sample_rate, num_subplots=10, howto100_feature_path=None):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes

        self.actions_dict = actions_dict
        self.actions_dict_rev = rrev_dict
        self.gt_path = gt_path
        self.features_path = features_path
        if howto100_feature_path is not None:
            self.howto100_feature_path = howto100_feature_path
        else:
            self.howto100_feature_path = features_path
        
        self.sample_rate = sample_rate
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        plt.axis('off')
        colors_txt = open(color_path, 'r').read().split('\n')[:-1]
        colors = []
        for i in range(len(colors_txt)):
            c = [float(v) for v in colors_txt[i].split(', ')]
            colors.append(c)
        self.colors = np.array(colors)
        # self.fig, self.ax = plt.subplots(num_subplots, 1, figsize=(20,2*num_subplots))
    
    def reset_fig(self, cur_subplots):
        self.fig, self.ax = plt.subplots(cur_subplots+1, 1, figsize=(20,2*(cur_subplots+1)))

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        # self.list_of_examples = \
        #     [p for p in self.list_of_examples if p.split("_")[0] == 'P26']
        no_p11 = []
        for vid in self.list_of_examples:
            if vid.split("_")[0] != 'P11':
                no_p11.append(vid)
        self.list_of_examples = no_p11
        # for p in self.list_of_examples:
        #     print(p)
        self.check_example_exist()
        random.shuffle(self.list_of_examples)
        
    
    def check_example_exist(self):
        print("=> Checking all examples exist in root_dir...")
        print("Number of samples pre-check: ", len(self.list_of_examples))
        cleaned = []
        for vid in self.list_of_examples:
            p_id = vid.split("_")[0]

            feature_filename = os.path.join(self.features_path, p_id, '{}.npy'.format(vid.split('.')[0]))
            if not os.path.exists(feature_filename):
                print("ERROR: {} does not exist".format(feature_filename))
                continue
            
            gt_filename = os.path.join(self.gt_path, '{}.txt'.format(vid))
            file_ptr = open(gt_filename, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(len(content))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            if np.all(classes == self.num_classes - 1): continue # All background

            cleaned.append(vid)

        self.list_of_examples = cleaned
        print("Number of samples post-check: ", len(self.list_of_examples))

    
    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size
        
        batch_input = []
        batch_target = []
        f_2D = None 
        f_3D = None
        for vid in batch:
            p_id = vid.split("_")[0]
            feature_filename = os.path.join(self.features_path, p_id, '{}.npy'.format(vid.split('.')[0]))
            f_3D = np.load(os.path.join(self.howto100_feature_path, '3D', p_id, '{}.npy'.format(vid.split('.')[0])))
            f_2D = np.load(os.path.join(self.howto100_feature_path, '2D', p_id, '{}.npy'.format(vid.split('.')[0])))
            
            features = np.load(feature_filename)
            features = features.T
            gt_filename = os.path.join(self.gt_path, '{}.txt'.format(vid))
            file_ptr = open(gt_filename, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(features.shape[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            batch_input .append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = [len(seq) for seq in batch_target]
        batch_input_tensor = torch.zeros(len(batch_input), batch_input[0].shape[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :batch_input[i].shape[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :batch_target[i].shape[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :batch_target[i].shape[0]] = torch.ones(self.num_classes, batch_target[i].shape[0])
        
        output_dict = dict()
        output_dict['batch_input_tensor'] = batch_input_tensor
        output_dict['batch_target_tensor'] = batch_target_tensor
        output_dict['mask'] = mask
        output_dict['batch'] = batch
        output_dict['f_2D'] = f_2D
        output_dict['f_3D'] = f_3D

        return output_dict
