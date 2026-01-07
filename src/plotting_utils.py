# src/plotting_utils.py

import matplotlib.pyplot as plt


def save_plot(fig, path):
    """
    Save matplotlib figure.
    """
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
