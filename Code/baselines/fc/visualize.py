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
    ax.text(-2.2, 0.25, cap, fontsize=30)
    ax.autoscale()


def visualize(config, ax, label, cap, filename=None):
    plot_color_bar(ax, label, config.colors, cap)
    if cap == "GT":
        plt.savefig(filename+'.png', dpi=300)
