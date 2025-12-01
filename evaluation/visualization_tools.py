# evaluation/visualization_tools.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def summarize_metrics(df: pd.DataFrame):
    """
    Print mean/std for the main metrics we care about.
    """
    print("Episodes:", len(df))

    for col in [
        "total_ore_value",
        "energy_used",
        "energy_efficiency",
        "collapse",
        "unique_tiles_visited",
        "ores_mined",
        "rocks_mined",
        "max_floor",
    ]:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            print(f"{col}: mean = {mean:.3f}, std = {std:.3f}")


def plot_hist(df: pd.DataFrame, column: str, bins: int = 20):
    """
    Plot a simple histogram for one metric.
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return

    plt.figure()
    plt.hist(df[column].dropna(), bins=bins)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(f"Histogram of {column}")
    plt.show()


def plot_scatter(df: pd.DataFrame, x: str, y: str):
    """
    Scatter plot for relationships, e.g. energy_used vs total_ore_value.
    """
    if x not in df.columns or y not in df.columns:
        print(f"Columns '{x}' or '{y}' not found.")
        return

    plt.figure()
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{y} vs {x}")
    plt.show()
