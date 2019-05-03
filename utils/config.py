import json
import os
from dotmap import DotMap
from sklearn.utils import class_weight
import numpy as np
import sys

def get_config_from_json(json_file):
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = DotMap(config_dict)
    return config


def process_config(args):
    path_json_file_model     =   args.__dict__['config']
    config                   =   get_config_from_json(path_json_file_model)

    if config.exp.name not in ["RGB", "Persistence", "Combined"]:
        print 'Specify model exp name ("RGB", "Persistence", "Combined") in config file : %s' % (path_json_file_model)
	sys.exit(0)

    config['data_path']      =   args.__dict__['data']
    config['findLR']         =  (args.__dict__['findLR']=='True')
    config['rgb']            =  (config.exp.name == 'RGB')         #(args.__dict__['rgb']=='True')
    config['custom']         =  (args.__dict__['custom']=='True')
    config['combined']       =  (config.exp.name == 'Combined')    #(args.__dict__['combined']=='True')
    config['set_weights']    =  (args.__dict__['set_weights']=='True')
    config['use_focal_loss'] =  (args.__dict__['use_focal_loss']=='True')


    config['results'] = args.__dict__['results']

    config['load_checkpoint'] = ''
    config['checkpoint_RGB'] = ''
    config['checkpoint_Persistence'] = ''

    config['checkpoint_dir'] = ''
    config['LRFinder']['save_dir'] = ''

    _, _, files_malignant_train = next(os.walk( os.path.join(config['data_path'], 'train', 'malignant', 'rgb') ))
    _, _, files_benign_train = next(os.walk( os.path.join(config['data_path'],'train', 'benign', 'rgb') ))
    _, _, files_malignant_cv = next(os.walk( os.path.join(config['data_path'],'val', 'malignant', 'rgb') ))
    _, _, files_benign_cv = next(os.walk( os.path.join(config['data_path'],'val', 'benign', 'rgb') ))
    _, _, files_malignant_test = next(os.walk( os.path.join(config['data_path'],'test', 'malignant', 'rgb') ))
    _, _, files_benign_test = next(os.walk( os.path.join(config['data_path'], 'test', 'benign', 'rgb') ))



    config['len_train'] = len(files_malignant_train) + len(files_benign_train)
    config['len_CV'] = len(files_malignant_cv) + len(files_benign_cv)
    config['len_test'] = len(files_malignant_test) + len(files_benign_test)



    if len(args.__dict__['load_checkpoint']) != 0:
        config['load_checkpoint'] = args.__dict__['load_checkpoint']

    if len(args.__dict__['checkpoint_RGB']) != 0:
        config['checkpoint_RGB'] = args.__dict__['checkpoint_RGB']

    if len(args.__dict__['checkpoint_Persistence']) != 0:
        config['checkpoint_Persistence'] = args.__dict__['checkpoint_Persistence']



    y_temp = [config['label']['malignant']] * len(files_malignant_train) + [config['label']['benign']] * len(files_benign_train)
    y_temp = np.array(y_temp)
    if config.set_weights:
        config['class_weights'] = class_weight.compute_class_weight('balanced', np.unique(y_temp), y_temp)
    else:
        config['class_weights'] = np.array([1.0, 1.0])

    config['config_path'] = args.__dict__['config']

    config = DotMap(config)

    if config.tensorboard.embeddings_layer_names == 'None':
        config.tensorboard.embeddings_layer_names = None

    if config.tensorboard.embeddings_metadata == 'None':
        config.tensorboard.embeddings_metadata = None

    if config.tensorboard.embeddings_data == 'None':
        config.tensorboard.embeddings_data = None

    
    if not os.path.isdir(config.results):
        os.makedirs(config.results)


    config.callbacks.tensorboard_log_dir = os.path.join(config.results, 'tensorboard')

    if not os.path.isdir(config.callbacks.tensorboard_log_dir):
        os.mkdir( os.path.join(config.results, 'tensorboard') )
        os.mkdir( os.path.join(config.results, 'tensorboard', 'training') )
        os.mkdir( os.path.join(config.results, 'tensorboard', 'validation') )
        print 'Created directory for tensorboard callback : %s' % (config.callbacks.tensorboard_log_dir)



    config.checkpoint_dir = os.path.join(config.results, 'checkpoint')
    if not os.path.isdir(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)
        print 'Created checkpoint directory : %s' % (config.checkpoint_dir)

    config.LRFinder.save_dir = os.path.join(config.results, 'LRFinder')
    if not os.path.isdir(config.LRFinder.save_dir):
        os.mkdir(config.LRFinder.save_dir)
        print 'Created LRFinder directory : %s' % (config.LRFinder.save_dir)


    return config








def process_config_test(args):
    path_json_file_model = os.path.join(args.__dict__['config_dir'], 'model.config')
    config = get_config_from_json(path_json_file_model)

    config['test_dir'] = args.__dict__['test_dir']

    if config.exp.name not in ["RGB", "Persistence", "Combined"]:
        print 'Specify model exp name ("RGB", "Persistence", "Combined") in config file : %s' % (path_json_file_model)
        sys.exit(0)

    config['rgb'] = ( config.exp.name == "RGB" )
    config['combined'] = ( config.exp.name == "Combined" )

    _, _, files_malignant_test = next(os.walk( os.path.join(config['test_dir'], 'malignant', 'rgb') ))
    _, _, files_benign_test = next(os.walk( os.path.join(config['test_dir'], 'benign', 'rgb') ))

    config['len_test'] = len(files_malignant_test) + len(files_benign_test)


    checkpoint_path, _, checkpoint_files = next(os.walk( os.path.join( args.__dict__['config_dir'], 'checkpoint' ) ))
    checkpoint_files = sorted(checkpoint_files)
    print 'Choosing best checkpoint : %s' % checkpoint_files[-1]
    config['load_checkpoint'] = os.path.join(checkpoint_path, checkpoint_files[-1])
    config['config_dir'] = args.__dict__['config_dir']

    config = DotMap(config)



    return config
