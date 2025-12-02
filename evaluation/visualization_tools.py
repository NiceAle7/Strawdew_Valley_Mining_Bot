# visualization/plot_utils.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#
# Reward curve from 1 CSV (SB3 Monitor compatible)
#
def plot_reward_curve(csv_path):
    # Skip SB3 metadata lines starting with #
    df = pd.read_csv(csv_path, comment='#')

    if 'r' not in df.columns:
        raise ValueError(f"Expected column 'r' not found in {csv_path}. Columns are: {df.columns}")

    plt.figure()
    plt.plot(df["r"].values)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Reward Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#
# Training curve (possibly filtered by algorithm column)
#
def plot_training_curves(results_csv, algo_name=None):
    df = pd.read_csv(results_csv, comment='#')

    if 'r' not in df.columns:
        raise ValueError(f"Expected column 'r' not found in {results_csv}. Columns are: {df.columns}")

    if algo_name and "algo" in df.columns:
        df = df[df["algo"] == algo_name]

    if "episode" in df.columns:
        grouped = df.groupby("episode")["r"].mean()
        values = grouped.values
    else:
        values = df["r"].values

    plt.figure()
    plt.plot(values)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Training Curve{' ('+algo_name+')' if algo_name else ''}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#
# Heatmap plot
#
def plot_heatmap(visit_counts, title="Visit Heatmap"):
    plt.figure()
    plt.imshow(visit_counts, origin="lower", interpolation="nearest")
    plt.colorbar(label="Visit Count")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


#
# Generic bar plot
#
def plot_bar(x, heights, xlabel="", ylabel="", title=""):
    plt.figure()
    plt.bar(x, heights)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()
