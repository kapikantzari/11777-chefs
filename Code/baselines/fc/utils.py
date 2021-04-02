#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
import argparse


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=[97]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=[97]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=[97]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
    
    if len(y_label) == 0:
        return 0, 0, 0

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def all_eval_scores(list_of_videos, gt_labels, recog_results, bg_class = [97], print_results = False):
    '''
    INPUT:
        list_of_videos contains a list of video_ids to evaluate
        gt_labels and recog_results are dictionaries mapping video_ids to list of gt_label or recog_result for that video
        if print_results is True, print all results in addition to return 
    RETURN: a dictionary mapping 'acc', 'edit', 'F1@xxx' to scores
    '''

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    results = dict()

    for vid in list_of_videos:
        gt_content = gt_labels[vid]

        recog_content = recog_results[vid]

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content, bg_class)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s], bg_class)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    results['acc'] = 100*float(correct)/total
    results['edit'] = (1.0*edit)/len(list_of_videos)

    if print_results:
        print("Acc: %.4f" % (100*float(correct)/total))
        print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
    
    # acc = (100*float(correct)/total)
    # edit = ((1.0*edit)/len(list_of_videos))
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])

        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100

        results['F1@{:.2f}'.format(overlap[s])] = f1

        if print_results:
            print('F1@%0.2f: %.4f' % (overlap[s], f1))

    return results

def single_eval_scores(gt_content, recog_content, bg_class = [97], print_results = False):
    '''
    INPUT:
        gt_content and recog_content are lists of ground truth and prediction labels for one video
        if print_results is True, print all results in addition to return 
    RETURN: a dictionary mapping 'acc', 'edit', 'F1@xxx' to scores
    '''
    list_of_videos = [0]
    gt_labels = {0:gt_content}
    recog_results = {0:recog_content}
    return all_eval_scores(list_of_videos, gt_labels, recog_results, bg_class, print_results)

# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks
def remap_labels(Y_all, old_bg_class = 97):
    # Map arbitrary set of labels (e.g. {1,3,5}) to contiguous sequence (e.g. {0,1,2})
    ys = np.unique([np.hstack([np.unique(Y_all[i]) for i in range(len(Y_all))])])
    y_max = ys.max() 
    y_map = np.zeros(y_max+1, np.int)-1
    new_bg_class = 0
    for i, yi in enumerate(ys):
        y_map[yi] = i
        new_bg_class = max(new_bg_class, i)
    Y_all = [y_map[Y_all[i]] for i in range(len(Y_all))]

    if y_map[old_bg_class] != -1:
        new_bg_class = y_map[old_bg_class]
    else:
        new_bg_class += 1

    return Y_all, new_bg_class