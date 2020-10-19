import os

from got10k.datasets import GOT10k
from siamfc import TrackerSiamFC

if __name__ == '__main__':
    root_dir = os.path.expanduser('~/data/GOT-10k')
    seqs = GOT10k(root_dir, subset='train', return_meta=True)

    tracker = TrackerSiamFC()
    tracker.train_over(seqs)
