import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_color_bar(ax, y, colors, y_start):
    x = [0]
    for i in range(len(y)-1):
        if y[i] != y[i+1]:
            x.append(i+1)
    x.append(len(y))
    height = 50
    width = 1500
    x_start = 100

    for i in range(len(x)-1):
        w = (x[i+1]-x[i]) / len(y) * width
        ax.add_patch(matplotlib.patches.Rectangle(
            (x_start, y_start), w, height, color=colors[y[x[i]]]))
        x_start += w


def visualize(config, ax, label, epoch, cap, filename=None):
    # y, y_hat are the ground truth and predicted labels of N frames of a video
    # filename: include some information about parameters
    
    plot_color_bar(ax, label, config.colors, -100*epoch)
    plt.text(-500, 20-100*epoch, cap, fontsize=12)
    
    if cap == "GT":
        plt.tight_layout()
        plt.savefig(filename+'.png', dpi=300)
