import os
import matplotlib.pyplot as plt

def save_plot(fig, filename):
    try:
        os.makedirs('charts', exist_ok=True)
        fig.savefig(f'charts/{filename}', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"Error in save_plot: {e}")
