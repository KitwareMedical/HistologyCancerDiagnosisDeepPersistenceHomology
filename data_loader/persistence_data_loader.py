import numpy as np
import skimage.io
import os, re, glob, random
import matplotlib.pyplot as plt
import cPickle as pickle
from keras.utils import to_categorical, Sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.applications.resnet50 import preprocess_input as preprocess_resnet
import histomicstk.preprocessing.color_normalization as htk_cnorm
from shutil import copyfile


class PersistenceBaseGenerator(Sequence):
    def __init__(self, config):

        self.config = config

        self.path_mal_train, _, self.files_malignant_train = next(os.walk( os.path.join(self.config['data_path'], 'train', 'malignant', 'persistence_images') ))
        self.path_ben_train, _, self.files_benign_train = next(os.walk( os.path.join(self.config['data_path'], 'train', 'benign',  'persistence_images') ))
        self.path_mal_cv, _, self.files_malignant_cv = next(os.walk( os.path.join(self.config['data_path'], 'val', 'malignant', 'persistence_images') ))
        self.path_ben_cv, _, self.files_benign_cv = next(os.walk( os.path.join(self.config['data_path'], 'val', 'benign', 'persistence_images') ))

        self.batch_size = self.config.trainer.batch_size
        self.label = self.config.label

        self.mal_paths_train = glob.glob( os.path.join(self.path_mal_train, '*') )
        self.ben_paths_train = glob.glob( os.path.join(self.path_ben_train, '*') )
        self.mal_paths_cv = glob.glob( os.path.join(self.path_mal_cv, '*') )
        self.ben_paths_cv = glob.glob( os.path.join(self.path_ben_cv, '*') )

        self.mal_outputs_train = [self.label['malignant']] * len(self.mal_paths_train)
        self.ben_outputs_train = [self.label['benign']] * len(self.ben_paths_train)
        self.mal_outputs_cv = [self.label['malignant']] * len(self.mal_paths_cv)
        self.ben_outputs_cv = [self.label['benign']] * len(self.ben_paths_cv)

        self.train_paths = self.mal_paths_train + self.ben_paths_train
        self.train_outputs = self.mal_outputs_train + self.ben_outputs_train
        self.cv_paths = self.mal_paths_cv + self.ben_paths_cv
        self.cv_outputs = self.mal_outputs_cv + self.ben_outputs_cv

        z = zip(self.train_paths, self.train_outputs)
        random.shuffle(z)
        self.train_paths, self.train_outputs = zip(*z)

        self.train_paths = self.train_paths[0 : self.config.len_train]
        self.train_outputs = self.train_outputs[0 : self.config.len_train]
        self.cv_paths = self.cv_paths[0 : self.config.len_CV]
        self.cv_outputs = self.cv_outputs[0 : self.config.len_CV]

        #self.up_max = 0.004191742773320068
        #self.lp_min =  5e-200


    def __len__(self):
        raise NotImplementedError


    def __getitem__(self, idx):
        raise NotImplementedError


    def on_epoch_end(self):
        z = zip(self.train_paths, self.train_outputs)
        random.shuffle(z)
        self.train_paths, self.train_outputs = zip(*z)


    def preprocess(self, img):
        img = 1.0*img/self.config.trainer.percentile_factor
        img = np.array([img])
        img = np.moveaxis(img, 0, 2)
        return img




class PersistenceTrainGenerator(PersistenceBaseGenerator):

    def __init__(self, config):
        super(PersistenceTrainGenerator, self).__init__(config)

        src = self.config.config_path
        dest = os.path.join(self.config.results, 'model.config')
        copyfile(src, dest)
        print 'Copied config file'
        print 'Train Length : ', len(self.train_paths)


    def __len__(self):
        return len(self.train_paths) / self.batch_size


    def __getitem__(self, idx):

        batchx = self.train_paths[ idx*self.batch_size : (idx+1)*self.batch_size ]
        batchy = self.train_outputs[ idx*self.batch_size : (idx+1)*self.batch_size ]

        X, Y = [], []

        for i in range(len(batchx)):
            path = batchx[i]
            with open(path, 'rb') as f:
                img = pickle.load(f)

            img = self.preprocess(img)

            X.append(img)
            Y.append( to_categorical(batchy[i], num_classes=2) )


        X = np.array(X)
        Y = np.array(Y)

        return (X, Y)






class PersistenceCVGenerator(PersistenceBaseGenerator):

    def __init__(self, config):
        super(PersistenceCVGenerator, self).__init__(config)
        print 'CV Length : ', len(self.cv_paths)

    def __len__(self):
        return len(self.cv_paths) / self.batch_size

    def __getitem__(self, idx):

        batchx = self.cv_paths[ idx*self.batch_size : (idx+1)*self.batch_size ]
        batchy = self.cv_outputs[ idx*self.batch_size : (idx+1)*self.batch_size ]

        X, Y = [], []

        for i in range(len(batchx)):
            path = batchx[i]
            with open(path, 'rb') as f:
                img = pickle.load(f)

            img = self.preprocess(img)

            X.append(img)
            Y.append( to_categorical(batchy[i], num_classes=2) )

        X = np.array(X)
        Y = np.array(Y)

        return (X, Y)



def PersistenceTestData(config):

    path_mal_test, _, files_malignant_test = next(os.walk( os.path.join(config.test_dir, 'malignant', 'persistence_images') ))
    path_ben_test, _, files_benign_test = next(os.walk( os.path.join(config.test_dir, 'benign', 'persistence_images') ))

    mal_paths_test = glob.glob( os.path.join(path_mal_test, '*') )
    ben_paths_test = glob.glob( os.path.join(path_ben_test, '*') )

    mal_outputs_test = [config.label.malignant] * len(mal_paths_test)
    ben_outputs_test = [config.label.benign] * len(ben_paths_test)

    test_paths = mal_paths_test + ben_paths_test
    test_outputs = mal_outputs_test + ben_outputs_test

    len_test = len(test_outputs)

    X = np.zeros((len_test, 32, 32, 1))
    Y = [-1]*len_test

    for i in range(len_test):
        path = test_paths[i]
        with open(path, 'rb') as f:
            img = pickle.load(f)

        img = 1.0*img/config.trainer.percentile_factor

        img = np.array([img])
     	img = np.moveaxis(img, 0, 2)
    	X[i] = img
        Y[i] = test_outputs[i]

    return (X, Y)
