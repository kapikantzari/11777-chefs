from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from args import get_args
import random
import os
from model import Net
from metrics import compute_metrics, print_computed_metrics, compute_epic_metrics
from loss import MaxMarginRankingLoss, TripletLoss
from gensim.models.keyedvectors import KeyedVectors
import pickle
from epic_dataloader import Epic_DataLoader

import wandb
wandb.login()
wandb.init(project='howto100m_feature_context_average_val', entity='chefs')

args = get_args()
config = wandb.config
config.clips_per_sample = args.clips_per_sample
config.lr = args.lr
config.epic_gt_verb = args.epic_gt_verb
config.clips_filter_length = args.clips_filter_length
config.clips_before = args.clips_before

if args.verbose:
    print(args)

# predefining random initial seeds
th.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.checkpoint_dir != '' and not(os.path.isdir(args.checkpoint_dir)):
    os.mkdir(args.checkpoint_dir)

if not(args.youcook) and not(args.msrvtt) and not(args.lsmdc) and not(args.epic):
    print('Loading captions: {}'.format(args.caption_path))
    caption = pickle.load(open(args.caption_path, 'rb'))
    print('done')

print('Loading word vectors: {}'.format(args.word2vec_path))
we = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
print('done')

if args.epic_gt_verb:
    gt_path = args.verb_gt_path
else:
    gt_path = args.narration_gt_path
args.features_path_2D = args.features_path_2D
args.features_path_3D = args.features_path_3D
start_idx = dict()
for vid in os.listdir(gt_path):
    gt = open(os.path.join(gt_path, vid), 'r').read().split('\n')[:-2]
    tmp = list(np.arange(len(gt)))
    tmp = np.array(list(filter(lambda x: (x==0 or (gt[x] != gt[x-1])), tmp)))
    start_idx[vid.strip('.txt')] = tmp

dataset = Epic_DataLoader(
    features_path = args.features_path_2D,
    features_path_3D = args.features_path_3D,
    start_idx = start_idx,
    gt_path = gt_path,
    we=we,
    we_dim=args.we_dim,
    max_words=args.max_words,
    clips_per_sample=args.clips_per_sample,
    clips_before=args.clips_before,
    clips_filter_length=args.clips_filter_length
)
dataset_size = len(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_thread_reader,
    shuffle=True,
    batch_sampler=None,
    drop_last=True,
)
print(len(dataloader))

dataloader_epic = DataLoader(
    dataset,
    batch_size=args.batch_size_val,
    num_workers=args.num_thread_reader,
    shuffle=False,
)
print("Length: ", len(dataloader_epic))

net = Net(
    video_dim=args.feature_dim,
    embd_dim=args.embd_dim,
    we_dim=args.we_dim,
    n_pair=args.n_pair,
    max_words=args.max_words,
    sentence_dim=args.sentence_dim,
)
net.train()
# Optimizers + Loss
if args.epic and args.epic_verb_only:
    loss_op = TripletLoss(
        margin=args.margin,
        negative_weighting=args.negative_weighting,
        batch_size=args.batch_size,
        n_pair=args.n_pair,
        hard_negative_rate=args.hard_negative_rate,
    )
else:
    loss_op = MaxMarginRankingLoss(
        margin=args.margin,
        negative_weighting=args.negative_weighting,
        batch_size=args.batch_size,
        n_pair=args.n_pair,
        hard_negative_rate=args.hard_negative_rate,
    )
net.cuda()
loss_op.cuda()

if args.pretrain_path != '':
    #args.pretrain_path = os.path.join('/raid/xiaoyuz1/EPIC', 'howto100m', 'model/howto100m_pt_model.pth')
    net.load_checkpoint(args.pretrain_path)

optimizer = optim.Adam(net.parameters(), lr=args.lr)

if args.verbose:
    print('Starting training loop ...')

def TrainOneBatch(model, opt, data, loss_fun, epic=True):
    text = data['text'].cuda()
    video = data['video'].cuda()
    video = video.view(-1, video.shape[-1])
    text = text.view(-1, text.shape[-2], text.shape[-1])
    opt.zero_grad()
    with th.set_grad_enabled(True):
        sim_matrix = model(video, text)
        if epic and args.epic_verb_only:
            labels = data['caption_cls'].cuda()
            loss = loss_fun(-sim_matrix, labels) * 1000
        else:
            loss = loss_fun(sim_matrix) #* 100
    loss.backward()
    opt.step()
    return loss.item()*100

def Eval_retrieval(model, eval_dataloader, dataset_name, cnt, epic=False):
    interval = len(eval_dataloader)
    model.eval()
    print('Evaluating Text-Video retrieval on {} data'.format(dataset_name))
    with th.no_grad():
        r1s = []
        r5s = []
        r10s = []
        mrs = []
        for i_batch, data in enumerate(eval_dataloader):
            text = data['text'].cuda()
            video = data['video'].cuda()
            m = model(video, text)
            m  = m.cpu().detach().numpy()
            if epic and args.epic_verb_only:
                metrics = compute_epic_metrics(m, data['caption_cls'].numpy().astype(int))
                metrics2 = compute_metrics(m.T)
                if (i_batch + 1) % interval == 0:
                    print_computed_metrics(metrics)
                    print_computed_metrics(metrics2)
            else:
                metrics = compute_metrics(m.T)
                r1 = metrics['R1']
                r5 = metrics['R5']
                r10 = metrics['R10']
                mr = metrics['MR']

                if len(text) >= args.batch_size_val * 0.5:
                    r1s.append(r1)
                    r5s.append(r5)
                    r10s.append(r10)
                    mrs.append(mr)
                
                print_computed_metrics(metrics)
        
        wandb_dict = {'val/R1': np.mean(r1s), 'val/R5': np.mean(r5s), 'val/R10':np.mean(r10s), 'val/MR':np.mean(mrs)}
        wandb.log(wandb_dict,step=cnt)

cnt = 0
for epoch in range(args.epochs):
    running_loss = 0.0
    if (epoch + 1) % args.eval_every == 0:
        Eval_retrieval(net, dataloader_epic, 'EpicKitchens',cnt,epic=True)
    if args.verbose:
        print('Epoch: %d' % epoch)
    
    for i_batch, sample_batch in enumerate(dataloader):
        batch_loss = TrainOneBatch(net, optimizer, sample_batch, loss_op, args.epic)
        wandb.log({'train/loss':batch_loss}, step=cnt)
        running_loss += batch_loss
        if (i_batch + 1) % args.n_display == 0 and args.verbose:
            print('Epoch %d, Epoch status: %.4f, Training loss: %.4f' %
            (epoch + 1, args.batch_size * float(i_batch) / dataset_size,
            running_loss / args.n_display))
            running_loss = 0.0
        cnt += 1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= args.lr_decay
    if args.checkpoint_dir != '' and (epoch+1)%10==0:
        path = os.path.join(args.checkpoint_dir, 'e{}.pth'.format(epoch + 1))
        net.save_checkpoint(path)

Eval_retrieval(net, dataloader_epic, 'EpicKitchens',cnt)
