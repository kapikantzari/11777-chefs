import os
import pickle
import numpy as np
from baseline import train, FC

import wandb

wandb.init(project='mstcn_v2', entity='chefs')

class Config:
    project_root = '/home/ubuntu'     #change me
    features_path = os.path.join(project_root, 'features')
    train_path = os.path.join(project_root, 'train.txt')
    validation_path = os.path.join(project_root, 'validation.txt')
    gt_path = os.path.join(project_root, 'groundTruth')
    action_dict_path = os.path.join(project_root, 'action_dictionary.pkl')
    results_path = os.path.join(project_root, 'results')

    model = "FC"
    model_by_name = {
        "FC":   FC,
    }
    class_label_map = pickle.load(open('action_dictionary.pkl', 'rb'))

    num_features = 1024
    num_classes = 98
    
    # Use fixed color mapping for better interpretability
    colors_txt = open('color.txt', 'r').read().split('\n')[:-1]
    colors = []
    for i in range(len(colors_txt)):
        c = [float(v) for v in colors_txt[i].split(', ')]
        colors.append(c)
    colors = np.array(colors)
    plot_freq = 10 #change this
    plot_samples = ['P16_04','P23_05','P29_05','P01_15','P05_07','P32_04','P26_39','P19_05','P01_11','P04_26','P07_17']
    
    batch_size = 16
    epochs = 50 #change this
    do_augment = True
    random_sampling = False #change this
    random_sampling_ratio = 100 #change this
    downsample = True
    # callbacks
    earlystopping_patience = 15
    reduce_lr_patience = 1
    reduce_lr_factor = 0.1
    minimum_lr = 1.0e-6
    lr = 1e-04

if __name__ == '__main__':
    train(Config)
