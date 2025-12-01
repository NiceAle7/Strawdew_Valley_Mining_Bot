import numpy as np
from evaluation.visualization_tools import plot_heatmap, plot_bar


# Bar plot example
x = ["Ore", "Rock", "Weeds"]
heights = [15, 8, 20]
plot_bar(x, heights, xlabel="Tile Type", ylabel="Count", title="Tiles Mined")
