
import argparse
import os

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('--findLR', default=False, help='Set this to True to run one epoch and save graph to find optimal LR')
    argparser.add_argument('--custom', default=False, help='Set this to True to train custom model')
    argparser.add_argument('--set_weights', default=False, help='Set this to True to use inverse class weights for unbalanced data')
    argparser.add_argument('--use_focal_loss', default=False, help='Set this to True to use focal loss')
    argparser.add_argument('--data', default='../data_new_split/', help='Path to data directory')
    argparser.add_argument('--config', default='configs/model_Persistence.config', help='Path to model config file')
    argparser.add_argument('--results', default='temp_results', help='Path to Results')
    argparser.add_argument('--load_checkpoint', default='', help='Path to hdf5 file to restore model from checkpoint')
    argparser.add_argument('--checkpoint_RGB', default='', help='Path to hdf5 file to restore model from checkpoint')
    argparser.add_argument('--checkpoint_Persistence', default='', help='Path to hdf5 file to restore model from checkpoint')
    argparser.add_argument('--test_dir', default='temp_results', help='Path to Test Directory')


    args = argparser.parse_args()
    return args
