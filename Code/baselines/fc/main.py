import os
import pickle
import numpy as np
from baseline import train, FC

import wandb

# wandb.init(project='mstcn', entity='chefs')

class Config:
    project_root = '/home/ubuntu'     #change me
    features_path = os.path.join(project_root, 'features')
    train_path = os.path.join(project_root, 'long_train.txt')
    validation_path = os.path.join(project_root, 'long_validation.txt')
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
    colors = np.random.rand(num_classes, 3)
    
    batch_size = 16
    epochs = 50
    do_augment = True
    random_sampling = False
    downsample = True
    # callbacks
    earlystopping_patience = 15
    reduce_lr_patience = 1
    reduce_lr_factor = 0.1
    minimum_lr = 1.0e-6
    lr = 3e-04

if __name__ == '__main__':
    train(Config)
