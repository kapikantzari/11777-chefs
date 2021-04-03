import pickle
import os
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', help="root directory of all data and annotations")
# parser.add_argument('--p_id', help="Participants id")

args = parser.parse_args()

class SampleItem(object):
    def __init__(self, p_id, root_dir):
        self.p_id = p_id 
        self.root_dir = root_dir 
        '''
        self.features:
            video_id ---> (num_features, 1024) frame features
        '''

        '''
        self.video_ids: list of video ids of the participant
        '''
        self.features, self.video_ids = self.get_all_features(root_dir)

        '''
        self.label_info:
            video_id ---> list of {'verb', 'noun', 'narration'} for each frame
        Tiffany Ma format
        '''
        self.label_info = self.load_label_info()

        self.all_verb_info = self.get_verbs()
    
    def get_all_features(self, root_dir):
        feat_dir = os.path.join(root_dir, 'features', self.p_id)
        video_ids = []
        if not os.path.isdir(feat_dir):
            print("ERROR! {} is not a feature directory for participant {}".format(feat_dir, p_id))
            return
        features = {}
        for ff in os.listdir(feat_dir):
            if ff.split(".")[-1] == "npy":
                filename = os.path.join(feat_dir, ff)
                d = np.load(filename)
                vid = ff.split(".")[0]
                features[vid] = d
                video_ids.append(vid)
        return features, video_ids
    
    def load_label_info(self):
        label_info = {}
        for vid in self.video_ids:
            label_dict = os.path.join(self.root_dir, 'label_dicts', '{}.pkl'.format(vid))
            with open(label_dict, 'rb') as f:
                data = pickle.load(f)
                label_info[vid] = data
        return label_info
    
    def get_verbs(self):
        all_verb_info = {}

        for i,vid in enumerate(self.video_ids):
            '''
            ########### WARNING ############3
            must -1 because of +1 bug in feature_extract_fmr
            '''
            num_feat = len(self.features[vid])-1

            v = []
            for j in range(num_feat):
                original_frame_idx = 4*j+1
                label_max_len = len(self.label_info[vid])
                if original_frame_idx >= label_max_len:
                    j_dict = {}
                    verb = 'background'
                else:
                    j_dict = self.label_info[vid][4*j+1]
                    if len(j_dict['verb']) == 0: 
                        verb = 'background'
                    else:
                        verb = j_dict['verb'][0]
                v.append(verb)

            all_verb_info[vid] = v
        
        return all_verb_info
    
    def output_to_file(self):
        gt_dir = os.path.join(self.root_dir, 'groundTruth')
        for vid in self.video_ids:
            filename = os.path.join(gt_dir, '{}.txt'.format(vid))

            if os.path.exists(filename):
                os.remove(filename)

            with open(filename, 'w+') as filehandle:
                for verb in self.all_verb_info[vid]:
                    filehandle.write('%s\n' % verb)


# sample = SampleItem(args.p_id, args.root_dir)
# sample.output_to_file()
feat_dir = os.path.join(args.root_dir, 'features')
for f in os.listdir(feat_dir):
    print(f)
    if f[0] == 'P':
        sample = SampleItem(f, args.root_dir)
        sample.output_to_file()
