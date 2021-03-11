import os
import os.path as osp
import mmcv
from got10k.datasets import OTB

dataset = OTB('/Users/Jerry/data/otb', download=False)

for s, (img_files, anno) in enumerate(dataset):
    seq_name = dataset.seq_names[s]
    print('--Sequence %d/%d: %s' % (s + 1, len(dataset), seq_name))
    if seq_name != 'Board':
        continue
    for img_file, box in zip(img_files, anno):
        box = box.copy()
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        mmcv.imshow_bboxes(img_file, box[None, :], show=False, colors='red',
                           thickness=2,
                           out_file=img_file.replace('otb', 'otb_gt'))
