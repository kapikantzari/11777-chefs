
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torch_xla
# import torch_xla.core.xla_model as xm

from data_loader import DataLoader
from utils import all_eval_scores
from visualize import visualize

import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = xm.xla_device()
print(device)

def evaluate(config, model, data, epoch, mode):
    y_hat_val = predict(data.X, model)
    gt_labels = dict()
    recog_results = dict()
    for i in range(len(data.video_ids)):
        gt_labels[data.video_ids[i]] = np.array(data.y[i])
        recog_results[data.video_ids[i]] = y_hat_val[i].reshape(-1)
    metrics = all_eval_scores(data.video_ids, gt_labels, recog_results, print_results=True)
    
    
    if (epoch+1) % 10 == 0 and mode == 'val':
        visualize(config, data.ax, y_hat_val[data.visualize_idx].reshape(-1), epoch//10, 'Epoch {} Pred'.format(epoch+1))
        if epoch == config.epochs - 1:
            visualize(config, data.ax, data.y[data.visualize_idx], epoch//10+1, "GT", filename=os.path.join(config.results_path, data.video_ids[data.visualize_idx]))
        
    wandb_dict = dict()
#     for key in metrics.keys():
#         wandb_dict[mode+"/"+key] = metrics[key]
#     wandb.log(wandb_dict, step=epoch)
    return metrics


def train(config):
    train_data = DataLoader(config, dataset='train')
    validation_data = DataLoader(config, dataset='validation')

    if config.model == 'FC':
        model = FC(config)
        print('Network:')
        print(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

        model = model.to(device)
        res = dict()
        for epoch in range(config.epochs):
            epoch_training_loss = 0
            for d in range(len(train_data.dataloaders)):
                for i, (inputs, labels) in enumerate(train_data.dataloaders[d], 0):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_training_loss += loss.item()
                epoch_training_loss /= len(train_data.y[d])
            
            res['[{}]:train'.format(epoch+1)] = evaluate(config, model, train_data, epoch, "train")
            res['[{}]:val'.format(epoch+1)] = evaluate(config, model, validation_data, epoch, "val")
            print('[{}] training loss: {}'.format(epoch + 1, epoch_training_loss))
            
        print('Finished Training')

        y_hat_val = predict(validation_data.X, model)
        train_accuracy = 0
        for i in range(len(y_hat_val)):
            train_acc = 1 - np.nonzero(validation_data.y[i]-y_hat_val[i].reshape(-1))[0].shape[0] / len(validation_data.y[i])
            train_accuracy += train_acc
        print('Network training accuracy: {:0.2f}'.format(train_accuracy/len(y_hat_val)))
       


        # gt_labels = dict()
        # recog_results = dict()
        # for i in range(validation_data.video_ids):
        #     gt_labels[validation_data.video_ids[i]] = validation_data.y[i]
        #     recog_results[validation_data.video_ids[i]] = y_hat_val[i]
        # metrics = all_eval_scores(validation_data.video_ids, gt_labels, recog_results, print_results=True)

        metric_txt = open(os.path.join(config.results_path,'{}_metrics.txt'.format(config.model)), 'w')
        metric_txt.write(str(res))
        metric_txt.close()
     

class FC(nn.Module):

    def __init__(self, config, train_mode = True):
        super(FC, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config.num_features,
                             self.config.num_features * 2)
        self.fc2 = nn.Linear(self.config.num_features * 2,
                             self.config.num_features * 4)
        self.fc3 = nn.Linear(self.config.num_features * 4,
                             self.config.num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def predict(X, model):
    res = []
    for i in range(len(X)):
        N = X[i].shape[0]
        tmp = model(torch.Tensor(X[i]).to('cuda')).detach().to('cpu').numpy()
        prediction = np.zeros((N,1),dtype=np.int64)
        for n in range(0,N):
            prediction[n,0] = np.argmax(tmp[n,:]) 
        res.append(prediction)
    return res

