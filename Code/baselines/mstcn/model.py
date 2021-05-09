#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import wandb
import pickle
from eval import single_eval_scores
from visualize import visualize, plot_table
import os 
import torch
import torch.nn.functional as F
from model_howto100m import Net 

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
            net = Net(
                video_dim=4096,
                embd_dim=6144,
                we_dim=300,
                n_pair=1,
                max_words=20,
                sentence_dim=-1,
            )
            pretrain_path = os.path.join(args.howto100m_model_dir)
            net.load_checkpoint(pretrain_path)
            self.howto100m_model = net.to(device)

            text_embeddings = []
            text_word_classes= []
            for fname in args.howto100m_text_dir:
                with open(fname, 'rb') as f:
                    embed, embed_verb_class = pickle.load(f) 
                    text_embeddings.append(embed)
                    text_word_classes.append(embed_verb_class)
            
            self.text_embeddings = torch.cat(text_embeddings, dim=0)
            self.text_word_classes = np.hstack(text_word_classes)
            print("Trainer: text_embeddings.shape", self.text_embeddings.shape, self.text_word_classes.shape)
        
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

    def predict_howto100m(self, predicted, f_2D, f_3D):
        predicted_np = predicted.detach().cpu().numpy().flatten()
        switch = np.where(predicted_np[1:] != predicted_np[:-1])[0]+1
        
        start_idx = np.append([0], switch)
        end_idx = np.append(switch, len(predicted_np))
        groups = np.hstack([start_idx.reshape(-1,1), end_idx.reshape(-1,1)])
        
        feat_3ds = []
        feat_2ds = []
        for start,end in groups:
            if end - start < 1:
                continue
            # print(len(f_2D))
            # print(int(np.floor(start*16/12)),int(np.ceil(end*16/12)))
            start_2d = int(np.floor(start*16/12))
            end_2d = int(np.ceil(end*16/12))
            if start_2d >= len(f_2D) or start >= len(f_3D):
                break
            
            feat_2d = np.amax(f_2D[start_2d:end_2d],axis=0).reshape((1,-1))
            feat_2ds.append(feat_2d)
            feat_3d = np.amax(f_3D[start:end],axis=0).reshape((1,-1))
            feat_3ds.append(feat_3d)
        
        segments_video_features = []
        for feat_2d, feat_3d in zip(feat_2ds, feat_3ds):
            feat_2d = F.normalize(torch.from_numpy(feat_2d).float(), dim=0)
            feat_3d = F.normalize(torch.from_numpy(feat_3d).float(), dim=0)
            segment = torch.cat((feat_2d, feat_3d), 1)
            segments_video_features.append(segment)
        video = torch.cat(segments_video_features, dim=0)
        video = video.cuda()
        video_feat = self.net.GU_video(video)
        sim_matrix = torch.matmul(video_feat, self.text_embeddings.t()) #row: each video | column: each 
        sim_matrix = sim_matrix.detach().cpu().numpy()
        top_100_phrases = np.argsort(sim_matrix, axis=1)[:,:100]
        votes = self.text_word_classes[top_100_phrases]

        def my_func(x):
            unique_class, counts = np.unique(x, return_counts=True)
            max_class = unique_class[np.argmax(counts)]
            return max_class

        predicted = np.apply_along_axis(my_func, 0, votes)
        print(predicted.shape)
        return predicted
    
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

                # if self.filter_background:
                #     mask_bkg = batch_target != self.background_class_idx
                # else:
                #     mask_bkg = batch_target >= 0
                mask_bkg = batch_target != self.background_class_idx
                _, predicted = torch.max(predictions[-1].data, 1)
                if self.args.use_howto100m:
                    predicted = self.predict_howto100m(predicted, f_3D, f_2D)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                correct_nobkg += ((predicted == batch_target).float()*mask[:, 0, :]*mask_bkg.squeeze(1)).sum().item()
                
                total += torch.sum(mask[:, 0, :]).item()
                if epoch >= num_epochs * 0.8:
                    results_dict = single_eval_scores(batch_target, predicted, bg_class = [self.num_classes-1])
                    f1_score += results_dict['F1@ 0.50']
                    edit_dist += results_dict['edit']

                if cnt % self.val_every == 0:
                    self.evaluate(val_batch_gen, num_epochs, epoch, cnt, device, batch_size)

                    wandb_dict = {'train/epoch_loss' : epoch_loss / batch_count, \
                        'train/acc' : float(correct)/total,
                        'train/acc_nobkg' : float(correct_nobkg)/total,
                        }
                    if epoch >= num_epochs * 0.8:
                        wandb_dict['train/edit'] = float(edit_dist) / batch_count
                        wandb_dict['train/F1'] = float(f1_score) / batch_count

                    #wandb.log(wandb_dict, step=cnt)

                cnt += 1
                
            scheduler.step()
            save_path = os.path.join(save_dir, '{}'.format(self.wandb_run_name))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            model_path = os.path.join(save_path, 'epoch-{}.model'.format(epoch+1))
            optimizer_path = os.path.join(save_path, 'epoch-{}.opt'.format(epoch+1))
            
            
            torch.save(self.model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)

            #wandb.save(model_path)
            #wandb.save(optimizer_path)
            
            print("Training: [epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples), float(correct)/total))

    def evaluate(self, val_batch_gen, num_epochs, epoch, cnt, batch_size):
        device = self.device
        self.model.eval()

        with torch.no_grad():
            correct = 0
            correct_nobkg = 0
            total = 0
            epoch_loss = 0
            f1_score = 0
            edit_dist = 0
            val_batch_gen.reset()

            while val_batch_gen.has_next():
                self.model.to(device)
                output_dict = val_batch_gen.next_batch(batch_size)
                batch_input = output_dict['batch_input_tensor'].to(device)
                batch_target = output_dict['batch_target_tensor'].to(device)
                mask = output_dict['mask'].to(device)
                f_3D = output_dict['f_3D']
                f_2D = output_dict['f_2D']
                batch_video_ids = output_dict['batch']

                predictions = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                mask_bkg = batch_target != self.background_class_idx
                _, predicted = torch.max(predictions[-1].data, 1)
                if self.args.use_howto100m:
                    predicted = self.predict_howto100m(predicted, f_3D, f_2D)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                correct_nobkg += ((predicted == batch_target).float()*mask[:, 0, :]*mask_bkg.squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

                torch.cuda.empty_cache()
                if epoch >= num_epochs * 0.8:
                    results_dict = single_eval_scores(batch_target, predicted, bg_class = [self.num_classes-1])
                    f1_score += results_dict['F1@ 0.50']
                    edit_dist += results_dict['edit']


                batch_video_id = batch_video_ids[0]
                if batch_video_id in self.videos_to_visualize: 
                    if (epoch+1) % self.visualize_every == 0:
                        
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
                        visualize(cnt, batch_video_id, batch_target, ax_name, fig_name, color_name, cap)
                    
                    plot_table(cnt, batch_video_id, predicted, batch_target, val_batch_gen.actions_dict_rev)

            wandb_dict = {'validate/epoch_loss' : epoch_loss / len(val_batch_gen.list_of_examples), \
                'validate/acc' : float(correct)/total,
                'validate/acc_nobkg' : float(correct_nobkg)/total,
                }
            if epoch >= num_epochs * 0.8:
                wandb_dict['validate/edit'] = float(edit_dist) / len(val_batch_gen.list_of_examples)
                wandb_dict['validate/F1'] = float(f1_score) / len(val_batch_gen.list_of_examples)

            #wandb.log(wandb_dict, step=cnt)
            print("Validate: [epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(val_batch_gen.list_of_examples), float(correct)/total))
    
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
