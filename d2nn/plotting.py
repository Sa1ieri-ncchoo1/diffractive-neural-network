"""Render saved metrics in a separate process to isolate plotting libraries."""
import csv
import sys
from pathlib import Path
import numpy as np

def plots(out, cm, history=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=7,
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    ax.set(xticks=range(10), yticks=range(10), xlabel='Predicted', ylabel='True',
           title=f'MNIST evaluation: {cm.trace()/cm.sum():.2%} ({cm.sum():,} samples)')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out/'confusion_matrix.png', dpi=150)
    plt.close(fig)
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for key, ax in [('accuracy', axes[0]), ('loss', axes[1])]:
            for split in ['train', 'test']:
                ax.plot([r['epoch'] for r in history], [r[f'{split}_{key}'] for r in history], label=split)
            ax.set(xlabel='Epoch', ylabel=key, title='New run (not historical thesis training)')
            ax.legend()
        fig.tight_layout()
        fig.savefig(out/'training_curves.png', dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    out=Path(sys.argv[1])
    cm=np.loadtxt(out/"confusion_matrix.csv",delimiter=",",dtype=int)
    history=None
    if (out/"history.csv").exists():
        with (out/"history.csv").open() as f:
            history=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    plots(out,cm,history)
