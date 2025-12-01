import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def plot_reward_curve(csv_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8,4))
    plt.plot(df['reward'].values)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Reward curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_training_curves(results_csv, algo_name=None):
    df = pd.read_csv(results_csv)
    if algo_name:
        df = df[df['algo']==algo_name]
    grouped = df.groupby('episode')['reward'].mean() if 'episode' in df.columns else df['reward']
    plt.figure(figsize=(8,4))
    plt.plot(grouped.values)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Training curve {algo_name if algo_name else ''}")
    plt.show()

def plot_heatmap(visit_counts, title="Visit heatmap"):
    """
    visit_counts: 2D numpy array of counts per tile
    """
    plt.figure(figsize=(6,6))
    plt.imshow(visit_counts, origin='lower', interpolation='nearest')
    plt.colorbar(label="Visit count")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def plot_bar(x, heights, xlabel="", ylabel="", title=""):
    plt.figure(figsize=(6,4))
    plt.bar(x, heights)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
