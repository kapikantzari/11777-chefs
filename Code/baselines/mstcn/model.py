#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import wandb
from eval import single_eval_scores
from visualize import visualize
import os 

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
    def __init__(self, wandb_run_name, num_blocks, num_layers, num_f_maps, dim, num_classes, val_every=50, visualize_every=5):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.loss_weight = torch.Tensor(loss_weight).to('cuda')
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

        self.val_every = val_every
        self.visualize_every = visualize_every
        self.wandb_run_name = wandb_run_name
        print("===> Trainer with wandb run named as: ", self.wandb_run_name)

        self.videos_to_visualize = ['P16_04','P23_05','P29_05','P01_15','P05_07','P32_04','P26_39','P19_05','P01_11','P04_26','P07_17']

    def train(self, save_dir, batch_gen, val_batch_gen, num_epochs, batch_size, learning_rate, device, \
        scheduler_step, scheduler_gamma):
        self.model.train()
        batch_gen.reset()
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

        cnt = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            batch_count = 0
            f1_score = 0
            edit_dist = 0
            batch_gen.reset()

            while batch_gen.has_next():
                self.model.to(device)
                batch_count += 1

                batch_input, batch_target, mask, _ = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()
                results_dict = single_eval_scores(batch_target, predicted, bg_class = [self.num_classes-1])
                f1_score += results_dict['F1@ 0.50']
                edit_dist += results_dict['edit']

                if cnt % self.val_every == 0:
                    self.evaluate(val_batch_gen, num_epochs, epoch, cnt, device, batch_size)

                    wandb_dict = {'train/epoch_loss' : epoch_loss / batch_count, \
                        'train/acc' : float(correct)/total,
                        'train/edit': float(edit_dist) / batch_count, \
                        'train/F1' : float(f1_score) / batch_count
                        }
                    wandb.log(wandb_dict, step=cnt)

                cnt += 1
                

            scheduler.step()
            save_path = os.path.join(save_dir, '{}'.format(self.wandb_run_name))
            if not os.file.exists(save_path):
                os.mkdir(save_path)
            model_path = os.path.join(save_path, 'epoch-{}.model'.format(epoch+1))
            optimizer_path = os.path.join(save_path, 'epoch-{}.opt'.format(epoch+1))
            
            
            torch.save(self.model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)

            wandb.save(model_path)
            wandb.save(optimizer_path)
            
            print("Training: [epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples), float(correct)/total))

    def evaluate(self, val_batch_gen, num_epochs, epoch, cnt, device, batch_size):
        self.model.eval()
        visualize_this_epoch = (epoch+1) % self.visualize_every == 0
        with torch.no_grad():
            correct = 0
            total = 0
            epoch_loss = 0
            f1_score = 0
            edit_dist = 0
            val_batch_gen.reset()

            while val_batch_gen.has_next():
                self.model.to(device)
                batch_input, batch_target, mask, batch_video_ids = val_batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)

                predictions = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

                torch.cuda.empty_cache()
                
                results_dict = single_eval_scores(batch_target, predicted, bg_class = [self.num_classes-1])
                f1_score += results_dict['F1@ 0.50']
                edit_dist += results_dict['edit']
                
                if not visualize_this_epoch:
                    continue

                batch_video_id = batch_video_ids[0]
                if batch_video_id in self.videos_to_visualize: 
                    if visualize_this_epoch:
                        ax_name = val_batch_gen.ax.flat[epoch//self.visualize_every]
                        color_name = val_batch_gen.colors
                        cap = 'Pred_Epoch_{}'.format(epoch)
                        visualize(batch_video_id, val_batch_gen.actions_dict_rev, predicted, ax_name, color_name, cap)
                    if epoch == num_epochs - 1:
                        ax_name = val_batch_gen.ax.flat[epoch//self.visualize_every]+1
                        color_name = val_batch_gen.colors
                        cap = 'GT'
                        visualize(batch_video_id, val_batch_gen.actions_dict_rev, batch_target, ax_name, color_name, cap)

            wandb_dict = {'validate/epoch_loss' : epoch_loss / len(val_batch_gen.list_of_examples), \
                'validate/acc' : float(correct)/total,
                'validate/edit' : float(edit_dist) / len(val_batch_gen.list_of_examples),
                'validate/F1' : float(f1_score) / len(val_batch_gen.list_of_examples),
                }
            wandb.log(wandb_dict, step=cnt)
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
