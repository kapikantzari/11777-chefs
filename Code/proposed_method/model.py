#!/usr/bin/python2.7

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
from numpy.core.defchararray import add
import wandb
import pickle
from eval import single_eval_scores
from visualize import visualize, plot_table
import os 
import torch
import torch.nn.functional as F
from model_howto100m import Net 
import pandas as pd 
from gensim.models.keyedvectors import KeyedVectors
import re

loss_weight = [106., 106., 106., 106., 106., 106., 106., 106., 106., 106., 106., \
       106., 106., 106., 106., 106., 106., 106., 106., 106., 106., 106., \
       106., 106., 106., 106., 106., 106., 106., 106., 106., 106., 106., \
       106., 106., 106., 106., 106., 106., 106., 106., 106., 106., 106., \
       106., 106., 106., 106., 106., 106., 106., 106., 106.,   2.]

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, args, wandb_run_name, num_classes, background_class_idx, device, val_every=50):
        
        self.args = args
        num_blocks = args.num_stages
        num_layers = args.num_layers
        num_f_maps = args.num_f_maps
        dim = args.features_dim
        visualize_every = args.visualize_every
        filter_background = args.filter_background
        self.device = device

        if self.args.use_howto100m:
            nar_path = os.path.join(args.root_dir, args.nar_path)
            vc_path = os.path.join(args.root_dir, args.verb_class_path)
            all_narrations = np.load(nar_path)
            labels = np.load(vc_path)
            print("Number of all_narration: ", len(all_narrations))
            
            net = Net(
                video_dim=4096,
                embd_dim=6144,
                we_dim=300,
                n_pair=1,
                max_words=20,
                sentence_dim=-1,
            )
            
            pretrain_path = os.path.join(args.root_dir, args.howto100m_cp)
            print('Loading howto100m: {}'.format(pretrain_path))
            net.load_checkpoint(pretrain_path)
            self.howto100m_model = net.to(device)
            print('done')

            self.howto100m_use_context = args.howto100m_use_context

            word2vec_path = os.path.join(args.root_dir, args.word2vec_cp)
            print('Loading word vectors: {}'.format(word2vec_path))
            we = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
            print('done')

            self.input_frames_per_feature = args.input_frames_per_feature
            self.howto100m_frames_per_feature = args.howto100m_frames_per_feature
            self.howto100m_ratio = self.howto100m_frames_per_feature /self.input_frames_per_feature
            self.howto100m_use_context = args.howto100m_use_context

            

            self.context_predictor = Context_Predictor(we, 300, 20, all_narrations, labels, self.howto100m_model, self.args.howto100m_use_context)
        
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.loss_weight = torch.Tensor(loss_weight).to('cuda')
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.background_class_idx = background_class_idx

        self.val_every = val_every
        self.visualize_every = visualize_every
        self.wandb_run_name = wandb_run_name
        print("===> Trainer with wandb run named as: ", self.wandb_run_name)

        self.videos_to_visualize = ['P16_04', 'P23_05', 'P29_05', 'P01_15','P05_07','P32_04','P26_39','P19_05','P01_11','P04_26','P04_32','P14_06','P19_05','P28_15','P07_17']

        self.filter_background = filter_background

        self.cur_subplots = 0

    def print_target(self, batch_target):
        batch_target_np = batch_target.detach().cpu().numpy().flatten()
        switch = np.where(batch_target_np[1:] != batch_target_np[:-1])[0]+1
        start_idx = np.append([0], switch)
        print("Target: ", ' | '.join([str(clas) for clas in batch_target_np[start_idx]]))

    
    def find_chunks(self, series):
        switch = np.where(series[1:] != series[:-1])[0]+1
        start_idx = np.append([0], switch)
        end_idx = np.append(switch, len(series))
        
        return pd.unique(series[start_idx])
    
    def predict_howto100m(self, predicted, bg_idx, f_3D, f_2D, batch_target, print_pred=False):
        predicted_np = predicted.detach().cpu().numpy().flatten()
        batch_target_np = batch_target.detach().cpu().numpy().flatten()
        switch = np.where(predicted_np[1:] != predicted_np[:-1])[0]+1

        start_idx = np.append([0], switch)
        end_idx = np.append(switch, len(predicted_np))
        original_groups = np.hstack([start_idx.reshape(-1,1), end_idx.reshape(-1,1)])
        groups_using_100m = (original_groups[:,1] - original_groups[:,0]) >= self.howto100m_ratio
        groups = original_groups[groups_using_100m]
        if len(groups) > 1 and print_pred:
            print("\n")
            for start,end in groups:
                unique_clas = self.find_chunks(batch_target_np[start:end])
                print(unique_clas)
        groups[:,0] = np.floor(groups[:,0] / self.howto100m_ratio)
        groups[:,1] = np.ceil(groups[:,1] / self.howto100m_ratio)
        if len(groups) <= 0:
            return torch.from_numpy(predicted_np).float().view((1,-1)) 

        start_2ds = np.floor(groups[:,0]*16/12).astype(int)
        assert np.all(start_2ds < len(f_2D))
        
        feat_3ds = []
        feat_2ds = []
        for start,end in groups:
            start_2d = int(np.floor(start*16/12))
            end_2d = int(np.ceil(end*16/12))
            feat_2d = np.amax(f_2D[start_2d:end_2d],axis=0).reshape((1,-1))
            feat_2ds.append(feat_2d)
            feat_3d = np.amax(f_3D[start:end],axis=0).reshape((1,-1))
            feat_3ds.append(feat_3d)

        # segments_video_features = []
        res = []
        prev = ""
        for i, (feat_2d, feat_3d) in enumerate(zip(feat_2ds, feat_3ds)):
            mstcn_label = int(predicted[0][i])
            if mstcn_label == bg_idx:
                res.append(bg_idx)
            else:
                feat_2d = F.normalize(torch.from_numpy(feat_2d).float(), dim=0)
                feat_3d = F.normalize(torch.from_numpy(feat_3d).float(), dim=0)
                segment = torch.cat((feat_2d, feat_3d), 1).cuda()
                # segments_video_features.append(segment)
                prev, label = self.context_predictor.predict(prev, segment)
                res.append(label)
        
        # video = torch.cat(segments_video_features, dim=0)
        # video = video.cuda()
        # video_feat = self.howto100m_model.GU_video(video)
        # sim_matrix = torch.matmul(video_feat, self.text_embeddings.t()) #row: each video | column: each 
        # sim_matrix = sim_matrix.detach().cpu().numpy()
        # top_100_phrases = np.argsort(sim_matrix, axis=1)[:,:10]
        # votes = self.text_word_classes[top_100_phrases]
        # if len(groups) > 1 and print_pred:
        #     print(votes)
        # def my_func(x):
        #     unique_class, counts = np.unique(x, return_counts=True)
        #     max_class = unique_class[np.argmax(counts)]
        #     return max_class

        # howto100m_predicted = np.apply_along_axis(my_func, 1, votes)

        howto100m_predicted = np.array(res)

        if len(groups) > 1 and print_pred:
            print(howto100m_predicted)
        # predicted_inflated = np.repeat(predicted, groups[:,1] - groups[:,0])
        # predicted_inflated = torch.from_numpy(predicted_inflated).float().view((1,-1))
        final_predicted = predicted_np.copy()
        seg_idx = 0
        for (orig_start, orig_end),seg_use100 in zip(original_groups, groups_using_100m):
            if not seg_use100:
                continue 
            final_predicted[orig_start:orig_end] = howto100m_predicted[seg_idx]
            seg_idx += 1
        
        final_predicted = torch.from_numpy(final_predicted).float().view((1,-1))
        return final_predicted
    
    def calculate_accuracy(self, predicted, batch_target, mask):
        mask_bkg = batch_target != self.background_class_idx
        num_correct = ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
        num_correct_bkg = ((predicted == batch_target).float()*mask[:, 0, :]*mask_bkg.squeeze(1)).sum().item()
        return num_correct, num_correct_bkg

    def train(self, save_dir, batch_gen, val_batch_gen, num_epochs, batch_size, learning_rate, \
        scheduler_step, scheduler_gamma):
        device = self.device
        self.model.train()
        batch_gen.reset()
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

        cnt = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            correct_nobkg = 0
            
            correct_howto100m = 0
            correct_howto100m_nobkg = 0
            
            total = 0
            batch_count = 0
            f1_score = 0
            edit_dist = 0
            f1_edit_count = 0
            batch_gen.reset()

            while batch_gen.has_next():
                self.model.to(device)
                batch_count += 1

                output_dict = batch_gen.next_batch(batch_size)
                batch_input = output_dict['batch_input_tensor'].to(device)
                batch_target = output_dict['batch_target_tensor'].to(device)
                mask = output_dict['mask'].to(device)
                f_3D = output_dict['f_3D']
                f_2D = output_dict['f_2D']
                
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                # Calculate accuracy not using howto100m:
                _, predicted_mstcn = torch.max(predictions[-1].data, 1)
                
                num_correct, num_correct_bkg = self.calculate_accuracy(predicted_mstcn, batch_target, mask)
                correct += num_correct
                correct_nobkg += num_correct_bkg
                
                if self.args.use_howto100m:
                    predicted = self.predict_howto100m(predicted_mstcn.clone(), self.background_class_idx, f_3D, f_2D, batch_target, print_pred=cnt >= 3000)
                    assert predicted.shape[1] == batch_target.shape[1]
                    predicted = predicted.to(device)
                    num_correct100, num_correct_bkg100 = self.calculate_accuracy(predicted, batch_target, mask)
                    correct_howto100m += num_correct100
                    correct_howto100m_nobkg += num_correct_bkg100
                else:
                    predicted = predicted_mstcn
                total += torch.sum(mask[:, 0, :]).item()
                if epoch >= num_epochs * 0.8:
                    results_dict = single_eval_scores(batch_target, predicted, bg_class = [self.num_classes-1])
                    f1_score += results_dict['F1@ 0.50']
                    edit_dist += results_dict['edit']

                if cnt % self.val_every == 0:
                    self.evaluate(val_batch_gen, num_epochs, epoch, cnt, batch_size)

                    wandb_dict = {'train/epoch_loss' : epoch_loss / batch_count, \
                        'train/acc' : float(correct)/total,
                        'train/acc_nobkg' : float(correct_nobkg)/total,
                        'train/acc_howto100m' : float(correct_howto100m)/total,
                        'train/acc_howto100m_nobkg' : float(correct_howto100m_nobkg)/total,
                        }
                    if epoch >= num_epochs * 0.8:
                        wandb_dict['train/edit'] = float(edit_dist) / batch_count
                        wandb_dict['train/F1'] = float(f1_score) / batch_count

                    if self.args.enable_wandb:
                        wandb.log(wandb_dict, step=cnt)

                cnt += 1

            scheduler.step()
            save_path = os.path.join(save_dir, '{}'.format(self.wandb_run_name))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            model_path = os.path.join(save_path, 'epoch-{}.pth'.format(epoch+1))
            optimizer_path = os.path.join(save_path, 'epoch-{}.opt'.format(epoch+1))
            
            
            torch.save(self.model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)

            print("Training: [epoch %d]: epoch loss = %f,   acc = %f,   howto100m_acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples), float(correct)/total, float(correct_howto100m)/total))

    def evaluate(self, val_batch_gen, num_epochs, epoch, cnt, batch_size):
        device = self.device
        self.model.eval()
        if self.args.howto100m_use_context:
            print("with context")
        else:
            print("no context")

        with torch.no_grad():
            correct = 0
            correct_nobkg = 0
            correct_howto100m = 0
            correct_howto100m_nobkg = 0
            total = 0
            epoch_loss = 0
            f1_score = 0
            edit_dist = 0
            f1_score_howto100m = 0
            edit_dist_howto100m = 0
            val_batch_gen.reset()

            while val_batch_gen.has_next():
                self.model.to(device)
                output_dict = val_batch_gen.next_batch(batch_size)
                batch_video_ids = output_dict['batch']
                batch_video_id = batch_video_ids[0]
                
                batch_input = output_dict['batch_input_tensor'].to(device)
                batch_target = output_dict['batch_target_tensor'].to(device)
                mask = output_dict['mask'].to(device)
                f_3D = output_dict['f_3D']
                f_2D = output_dict['f_2D']
                
                predictions = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                
                # Calculate accuracy not using howto100m:
                _, predicted_mstcn = torch.max(predictions[-1].data, 1)
                num_correct, num_correct_bkg = self.calculate_accuracy(predicted_mstcn, batch_target, mask)
                correct += num_correct
                correct_nobkg += num_correct_bkg
                results_dict = single_eval_scores(batch_target, predicted_mstcn.to(device), bg_class = [self.num_classes-1])
                f1_score += results_dict['F1@ 0.50']
                edit_dist += results_dict['edit']
                
                if self.args.use_howto100m:
                    predicted = self.predict_howto100m(predicted_mstcn.clone(), self.background_class_idx, f_3D, f_2D, batch_target, print_pred=False)
                    assert predicted.shape[1] == batch_target.shape[1]
                    predicted = predicted.to(device)
                    num_correct100, num_correct_bkg100 = self.calculate_accuracy(predicted, batch_target, mask)
                    correct_howto100m += num_correct100
                    correct_howto100m_nobkg += num_correct_bkg100
                else:
                    predicted = predicted_mstcn
                
                total += torch.sum(mask[:, 0, :]).item()
                torch.cuda.empty_cache()
                if epoch >= num_epochs * 0.8:
                    results_dict_howto100m = single_eval_scores(batch_target, predicted, bg_class = [self.num_classes-1])
                    f1_score_howto100m += results_dict_howto100m['F1@ 0.50']
                    edit_dist_howto100m += results_dict_howto100m['edit']


                batch_video_id = batch_video_ids[0]
                if batch_video_id in self.videos_to_visualize: 
                    if ((epoch+1) % self.visualize_every == 0) or (num_epochs == 0):
                        
                        self.cur_subplots += 1
                        val_batch_gen.reset_fig(1)
                        
                        ax_name = val_batch_gen.ax.flat[0]
                        fig_name =  val_batch_gen.fig
                        color_name = val_batch_gen.colors
                        cap = 'Pred_Epoch_{}'.format(epoch)
                        visualize(cnt, batch_video_id, predicted, ax_name, fig_name, color_name, cap)
                    
                        #if epoch == num_epochs - 1:
                        ax_name = val_batch_gen.ax.flat[1]
                        fig_name =  val_batch_gen.fig
                        color_name = val_batch_gen.colors
                        cap = 'GT'
                        visualize(cnt, batch_video_id, batch_target, ax_name, fig_name, color_name, cap, enable_wandb=self.args.enable_wandb)
                    
                    plot_table(cnt, batch_video_id, predicted, batch_target, val_batch_gen.actions_dict_rev, enable_wandb=self.args.enable_wandb)
            if num_epochs == 0 and epoch == 0:
                wandb_dict = {'train/epoch_loss' : epoch_loss / len(val_batch_gen.list_of_examples), \
                    'train/acc' : float(correct)/total,
                    'train/acc_nobkg' : float(correct_nobkg)/total,
                    'train/acc_howto100m' : float(correct_howto100m)/total,
                    'train/acc_howto100m_nobkg' : float(correct_howto100m_nobkg)/total,
                    }
                if epoch >= num_epochs * 0.8:
                    wandb_dict['train/edit'] = float(edit_dist) / len(val_batch_gen.list_of_examples)
                    wandb_dict['train/F1'] = float(f1_score) / len(val_batch_gen.list_of_examples)

                if self.args.enable_wandb:
                    wandb.log(wandb_dict, step=cnt)
                print("Training: [epoch %d]: epoch loss = %f,   acc = %f,   howto100m_acc = %f" % (epoch + 1, epoch_loss / len(val_batch_gen.list_of_examples), float(correct)/total, float(correct_howto100m)/total))
            else:
                wandb_dict = {'validate/epoch_loss' : epoch_loss / len(val_batch_gen.list_of_examples), \
                    'validate/acc' : float(correct)/total,
                    'validate/acc_nobkg' : float(correct_nobkg)/total,
                    'validate/acc_howto100m' : float(correct_howto100m)/total,
                    'validate/acc_howto100m_nobkg' : float(correct_howto100m_nobkg)/total,
                    }
                if epoch >= num_epochs * 0.8:
                    wandb_dict['validate/edit'] = float(edit_dist) / len(val_batch_gen.list_of_examples)
                    wandb_dict['validate/F1'] = float(f1_score) / len(val_batch_gen.list_of_examples)
                    wandb_dict['validate/edit_howto100m'] = float(edit_dist_howto100m) / len(val_batch_gen.list_of_examples)
                    wandb_dict['validate/F1_howto100m'] = float(f1_score_howto100m) / len(val_batch_gen.list_of_examples)
                if self.args.enable_wandb:
                    wandb.log(wandb_dict, step=cnt)
                print("Validate: [epoch %d]: epoch loss = %f,   acc = %f,   howto100m_acc = %f" % (epoch + 1, epoch_loss / len(val_batch_gen.list_of_examples), float(correct)/total, float(correct_howto100m)/total))
                print(wandb_dict)
    
    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [actions_dict.keys()[actions_dict.values().index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))

class Context_Predictor():
    def __init__(self, we, we_dim, max_words, all_narrations, # array of all narrations
            labels, # verb class of each narration in all_narrations
            net, #howto100m
            use_context,
    ):
        self.labels = labels
        self.all_narrations = all_narrations
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.net = net
        self.use_context = use_context
        if not use_context:
            all_text_features = []
            for t in self.all_narrations:
                all_text_features.append(self._words_to_we(self._tokenize_text(str(t))))
            self.all_text_features = torch.from_numpy(np.array(all_text_features, dtype=np.float32)).cuda()

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
#             return torch.from_numpy(we)
            return we
        else:
#             return torch.zeros(self.max_words, self.we_dim) 
            return np.zeros((self.max_words, self.we_dim))

    def _tokenize_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w    

    def predict(self, prev, video_feature):
        
        all_text_features = []
        if self.use_context:
            for t in self.all_narrations:
                all_text_features.append(self._words_to_we(self._tokenize_text(prev+' '+str(t))))
            all_text_features = torch.from_numpy(np.array(all_text_features, dtype=np.float32)).cuda()
        else:
            all_text_features = self.all_text_features
        total_len = len(all_text_features)
        split = 2
        split_len = total_len // split
        sim_vector = None
        for i in range(split):
            start = i*split_len
            end = total_len if (i+1==split) else ((i+1)*split_len)
            tmp = self.net(video_feature, all_text_features[start:end])
            if sim_vector == None:
                sim_vector = tmp
            else:
                sim_vector = torch.cat((sim_vector, tmp))
        prediction_top10 = (-sim_vector).argsort(axis=0)[:10].reshape(-1,)
        return self.all_narrations[prediction_top10[0]], self.labels[prediction_top10[0]]
