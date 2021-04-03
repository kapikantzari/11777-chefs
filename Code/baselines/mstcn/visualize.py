import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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
    labels = [actions_dict_rev[y[0]]]
    for i in range(len(y)-1):
        if y[i] != y[i+1]:
            x.append(i+1)
            labels.append(actions_dict_rev[y[i+1]])
    labels.append(actions_dict_rev[y[len(y)-1]])

    return labels

def visualize(epoch, cnt, batch_video_id, y, y_hat, ax, colors, actions_dict_rev):
    # y, y_hat are the ground truth and predicted labels of N frames of a video
    # filename: include some information about parameters
    y = y.cpu().numpy().reshape(-1)
    y_hat = y_hat.cpu().numpy().reshape(-1)
    
    plot_color_bar(ax, y, colors, 0)
    plt.text(-100, 20, "GT", fontsize=12)
    plot_color_bar(ax, y_hat, colors, -100)
    plt.text(-120, -80, "Pred", fontsize=12)
    
    plt.xlim([-200, 1100])
    plt.ylim([-300, 200])  
    
    plt.tight_layout()
    wandb.log({'image/{}_{}'.format(batch_video_id, epoch): plt}, step=cnt)
    # plt.savefig('/home/ubuntu/results/qualitative_result.png', dpi=300)
    action_str_list = table_list(y, actions_dict_rev)
    wandb.log({"table/pred_{}_{}".format(batch_video_id, epoch): wandb.Table(data=action_str_list, columns=["Predicted Label"])}, step=cnt)
    action_str_list = table_list(y_hat, actions_dict_rev)
    wandb.log({"table/gt_{}_{}".format(batch_video_id, epoch): wandb.Table(data=action_str_list, columns=["Groundtruth Label"])}, step=cnt)

    plt.close()

def visualize(batch_video_id, actions_dict_rev, y, ax, colors, cap, filename=None):
    # y, y_hat are the ground truth and predicted labels of N frames of a video
    # filename: include some information about parameters
    y = y.cpu().numpy().reshape(-1)
    plot_color_bar(ax, y, colors, cap)

    action_str_list = table_list(y, actions_dict_rev) 
    wandb.log({"table/{}_{}".format(batch_video_id, cap): wandb.Table(data=action_str_list, columns=["{} Label".format(cap)])}, step=cnt)

    if cap == "GT":
        plt.tight_layout()
        wandb.log({'image/{}'.format(batch_video_id): plt}, step=cnt)

