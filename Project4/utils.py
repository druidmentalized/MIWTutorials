import pandas as pd
from datetime import datetime

def plot_history(history, marker):
    df = pd.DataFrame(history.history)
    ax = df.plot()

    fig = ax.get_figure()
    timestamp = datetime.now().strftime("%d%m%Y")
    fig.savefig(f'history_plot_{marker}_{timestamp}.png')
