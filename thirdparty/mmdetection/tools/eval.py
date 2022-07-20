import argparse 
import pickle 
import mmcv 
from mmcv import Config, DictAction, PickleHandler 

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.datasets.coco import CocoDataset
import ipdb

def parse_args():
    parser = argparse.ArgumentParser(
        description='Eval Script'
    )
    parser.add_argument('pkl_results', help='detection results (pkl or pickle format)')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config) 
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True
    
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.pkl_results)
    
    metric = dataset.evaluate(
            outputs, 
            jsonfile_prefix='/home/SENSETIME/ligang2/Works/SSOD/checkpoints/results')
    print(metric)

if __name__=="__main__":
    main()