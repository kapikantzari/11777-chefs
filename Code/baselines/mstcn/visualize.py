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
    width = 1000
    x_start = 0

    for i in range(len(x)-1):
        w = (x[i+1]-x[i]) / len(y) * width
        ax.add_patch(matplotlib.patches.Rectangle(
            (x_start, y_start), w, height, color=colors[y[x[i]]]))
        x_start += w


def visualize(y, y_hat, ax, colors):
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
    plt.savefig('/home/ubuntu/results/qualitative_result.png', dpi=300)
