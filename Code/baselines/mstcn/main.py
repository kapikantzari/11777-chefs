#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
import pickle
import numpy as np


# Flexible integration for any Python script
import wandb

vocab_subset = {'cooking': [0, 1, 2, 7, 9, 10, 13, 14, 15, 16, 18, 19, 21, 22, 23, 25, 26, 28, 34, 35, 36, 39, 42, 43,  45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 63, 66, 69, 75, 76, 77, 80, 81, 82, 83, 84, 90, 92, 93, 95, 96], 'salad': [46, 7, 19, 1, 82, 10, 92]}
wandb.init(project='mstcn', entity='chefs')
wandb_run_name = wandb.run.name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
# parser.add_argument('--dataset', default="epic_kitchen")
# parser.add_argument('--split', default='1')

parser.add_argument('--root_dir', help="root directory of all data and annotations")
parser.add_argument('--background_name', default="background", help="what verb to call the background verb, default as background")

parser.add_argument('--num_stages', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=10)
parser.add_argument('--num_f_maps', type=int, default=64)
parser.add_argument('--features_dim', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--scheduler_step', type=int, default=15)
parser.add_argument('--scheduler_gamma', type=float, default=0.7)
parser.add_argument('--vocab_subset', type=str, default=None)
parser.add_argument('--visualize_every', type=int, default=5)

args = parser.parse_args()

num_stages = args.num_stages
num_layers = args.num_layers
num_f_maps = args.num_f_maps
features_dim = args.features_dim
bz = args.batch_size
lr = args.lr
num_epochs = args.num_epochs
sample_rate = 1
scheduler_step = args.scheduler_step
scheduler_gamma = args.scheduler_gamma
visualize_every = args.visualize_every
num_subplots = num_epochs // visualize_every + 1

config = wandb.config
config.num_stages = num_stages
config.num_layers = num_layers
config.num_f_maps = num_f_maps
config.features_dim = features_dim
config.bz = bz
config.lr = lr
config.num_epochs = num_epochs
config.sample_rate = sample_rate
config.scheduler_step = scheduler_step
config.scheduler_gamma = scheduler_gamma
config.visualize_every = visualize_every

# vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
# vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
# features_path = "./data/"+args.dataset+"/features/"
# gt_path = "./data/"+args.dataset+"/groundTruth/"

# mapping_file = "./data/"+args.dataset+"/mapping.txt"

# model_dir = "./models/"+args.dataset+"/split_"+args.split
# results_dir = "./results/"+args.dataset+"/split_"+args.split
 
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)

# file_ptr = open(mapping_file, 'r')
# actions = file_ptr.read().split('\n')[:-1]
# file_ptr.close()
# actions_dict = dict()
# for a in actions:
#     actions_dict[a.split()[1]] = int(a.split()[0])

# num_classes = len(actions_dict)

vid_list_file = os.path.join(args.root_dir, 'train.txt')
vid_list_file_tst = os.path.join(args.root_dir, 'validation.txt')
features_path = os.path.join(args.root_dir, 'features')
gt_path = os.path.join(args.root_dir, 'groundTruth')
color_path = os.path.join(args.root_dir, 'color.txt')

action_dict_file = os.path.join(args.root_dir, 'action_dictionary.pkl')
actions_dict = None
with open(action_dict_file, 'rb') as f:
    actions_dict = pickle.load(f)

rev_dict_file = os.path.join(args.root_dir, 'verb.txt')
rev_dict = {}
file_ptr = open(rev_dict_file, 'r')
action_name_list = file_ptr.read().split('\n')[:-1]
file_ptr.close()
for name_idx, name in enumerate(action_name_list):
    rev_dict[name] = name_idx

if args.vocab_subset != None:
    temp_map = {}
    for i, idx in enumerate(vocab_subset[args.vocab_subset]):
        temp_map[idx] = i

    max_val = np.max(list(temp_map.values()))
    bg_cls = max_val + 1

    for k, v in actions_dict.items():
        if v not in vocab_subset[args.vocab_subset]:
            actions_dict[k] = bg_cls
        else:
            actions_dict[k] = temp_map[v]
    actions_dict[args.background_name] = bg_cls

    for k, v in rev_dict.items():
        if v not in vocab_subset[args.vocab_subset]:
            rev_dict[k] = bg_cls
        else:
            rev_dict[k] = temp_map[v]
    rev_dict[args.background_name] = bg_cls

    rrev_dict = {}
    for k,v in rev_dict.items():
        rrev_dict[v] = k

else:
    max_val = np.max(list(actions_dict.values()))
    actions_dict[args.background_name] = max_val+1

    rev_dict[args.background_name] = max_val+1
    rrev_dict = {}
    for k,v in rev_dict.items():
        rrev_dict[v] = k

max_val = np.max(list(actions_dict.values()))
num_classes = max_val+1
print("Num classes: ", num_classes)

model_dir = os.path.join(args.root_dir, 'models')
results_dir = os.path.join(args.root_dir, 'results')

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

trainer = Trainer(wandb_run_name, num_stages, num_layers, num_f_maps, features_dim, num_classes, visualize_every=visualize_every)
if args.action == "train":
    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_gen = BatchGenerator(num_classes, actions_dict, rrev_dict, gt_path, features_path, color_path, sample_rate, num_subplots)
    batch_gen.read_data(vid_list_file)
    batch_gen.check_example_exist()

    val_batch_gen = BatchGenerator(num_classes, actions_dict, rrev_dict, gt_path, features_path, color_path, sample_rate, num_subplots)
    val_batch_gen.read_data(vid_list_file_tst)
    val_batch_gen.check_example_exist()
    
    trainer.train(model_dir, batch_gen, val_batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device, scheduler_step=scheduler_step, scheduler_gamma=scheduler_gamma)

# if args.action == "predict":
#     trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
