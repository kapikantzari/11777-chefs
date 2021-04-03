import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
import io
from PIL import Image
import PIL

def plot_color_bar(ax, y, colors, cap):
    ax.axis('off')
    x = [0]
    for i in range(len(y)-1):
        if y[i] != y[i+1]:
            x.append(i+1)
    x.append(len(y))
    height = 0.5
    width = 10
    x_start = 0

    for i in range(len(x)-1):
        w = (x[i+1]-x[i]) / len(y) * width
        ax.add_patch(matplotlib.patches.Rectangle(
            (x_start, 0), w, height, color=colors[y[x[i]]]))
        x_start += w
    ax.text(-2.5, 0.25, cap, fontsize=30)
    ax.autoscale()

def table_list(y, actions_dict_rev):
    y = y.view(-1,).detach().cpu().numpy()
    labels = [actions_dict_rev[y[0]]]
    for i in range(len(y)-1):
        if y[i] != y[i+1]:
            labels.append(actions_dict_rev[y[i+1]])

    return labels

def plot_table(cnt, batch_video_id, y_pred, y_gt, actions_dict_rev):
    alist_pred = table_list(y_pred, actions_dict_rev)
    alist_gt = table_list(y_gt, actions_dict_rev)

    l = max(len(alist_pred), len(alist_gt))
    alist_preds = [alist_pred[i] if i < len(alist_pred) else '---' for i in range(l)]
    alist_gts = [alist_gt[i] if i < len(alist_gt) else '---' for i in range(l)]
    idx = np.arange(l)+1
    data = np.array([idx, alist_preds, alist_gts]).T
    
    wandb.log({"table/{}".format(batch_video_id): wandb.Table(data=data, columns=["Index", "Predicted", "GT"])}, step=cnt)

def visualize(cnt, batch_video_id, y, ax, fig, colors, cap, filename=None):
    # y, y_hat are the ground truth and predicted labels of N frames of a video
    # filename: include some information about parameters
    y = y.cpu().numpy().reshape(-1)
    plot_color_bar(ax, y, colors, cap)

    if cap == "GT":
        # plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        wandb.log({'image/{}'.format(batch_video_id): wandb.Image(PIL.Image.open(buf))}, step=cnt)
        # wandb.log({'image/{}'.format(batch_video_id): plt}, step=cnt)
        plt.close()
