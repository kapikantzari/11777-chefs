from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import Dataset
import pickle
import torch.nn.functional as F
import numpy as np
import re
from collections import defaultdict
from torch.utils.data.dataloader import default_collate

import os


class Epic_DataLoader(Dataset):
    """Epic-Kitchens dataset loader"""

    def __init__(
            self,
            features_path,
            features_path_3D,
            start_idx,      # {video_id : array of start idx of a segmentation}
            gt_path,
            we,
            we_dim=300,
            max_words=10,
            seg_threshold=-1,
            train=True
    ):
        self.seg_threshold = seg_threshold
        self.data = self.__load_data(features_path, features_path_3D, start_idx, gt_path)
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words

    def __load_data(self, features_path, features_path_3D, start_idx, gt_path):
        data = []
        for id in start_idx.keys():
          if os.path.isfile(os.path.join(features_path, id[0:3]+'/'+id+'.npy')):
            f_2D = np.load(os.path.join(features_path, id[0:3]+'/'+id+'.npy'))     # (x, 2048)
            f_3D = np.load(os.path.join(features_path_3D, id[0:3]+'/'+id+'.npy'))  # (x, 2048)
            gt = open(os.path.join(gt_path, id+'.txt'), 'r').read().split('\n')[:-1]
            prev_caption = ''
            for i in range(len(start_idx[id])):
                start = start_idx[id][i]
                if i == len(start_idx[id]) - 1:
                    end = len(f_3D)
                else:
                    end = start_idx[id][i+1]
                # print(id, int(np.floor(start*16/12)), int(np.ceil(end*16/12)), start, end)
                if end - start > self.seg_threshold and gt[start] != 'background':
                    caption = prev_caption +' '+ gt[start]
                    prev_caption = gt[start]
                    data.append({'id': id, 'start': start, 'end': end, '2d': np.amax(f_2D[int(np.floor(start*16/12)):int(np.ceil(end*16/12))],axis=0).reshape((1,-1)), '3d': np.amax(f_3D[start:end],axis=0).reshape((1,-1)), 'caption': caption})
        return data

    def _zero_pad_tensor(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
            return np.concatenate((tensor, zero), axis=0)

    def _words_to_we(self, words):
        words = [word for word in words if word in self.we.vocab]
        if words:
            we = self._zero_pad_tensor(self.we[words], self.max_words)
            return th.from_numpy(we)
        else:
            return th.zeros(self.max_words, self.we_dim) 

    def _tokenize_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d']).float(), dim=0)
        feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d']).float(), dim=0)
        video = th.cat((feat_2d, feat_3d), 1)[0]
        cap = self.data[idx]['caption']
        caption = self._words_to_we(self._tokenize_text(cap))
        return {'video': video, 'text': caption, 'video_id': self.data[idx]['id'], 'start': self.data[idx]['start'], 'end': self.data[idx]['end']}
