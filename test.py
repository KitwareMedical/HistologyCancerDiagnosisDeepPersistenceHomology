from utils.config import process_config_test

from models.resnetRGB_model import ResNetRGBModel
from trainer.resnetRGB_trainer import ResNetRGBTrainer

from models.Persistence_model import PersistenceModel
from trainer.Persistence_trainer import PersistenceTrainer

from models.resnetCombined_model import ResNetCombinedModel
from trainer.resnetCombined_trainer import ResNetCombinedTrainer, ResNetCombinedTest

from utils.evaluate import test_combined, test_rgb, test_persistence

import numpy as np
from sklearn import metrics
import sys, os
import argparse


from data_loader.combined_data_loader import CombinedTestData
from keras import Model
import cPickle as pickle


def main():

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--test_dir', default='', help='Path to Test Directory')
    argparser.add_argument('--config_dir', default='', help='Path to Config Directory (Containing model checkpoints and config file)')

    args = argparser.parse_args()


    if args.__dict__['test_dir'] == '':
        print '-'*90
        print 'Provide path to test folder: --test_dir (Should have directories : malignant and benign)'
        print '-'*90
        sys.exit(0)

    if args.__dict__['config_dir'] == '':
        print '-'*90
        print 'Provide path to config folder: --config_dir'
        print '-'*90
        sys.exit(0)



    config = process_config_test(args)

    print '\n', config, '\n'


    if config.rgb:
        print 'Testing RGB model'
        test_rgb(config)

    elif config.combined:
        print 'Testing Combined model'
        test_combined(config)

    else:
        print 'Testing Persistence model'
        test_persistence(config)


if __name__ == "__main__":
    main()
