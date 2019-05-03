import numpy as np
import skimage.io
import os, sys, time
import re
import matplotlib.pyplot as plt
import cPickle as pickle
import h5py
import glob
from keras.utils import to_categorical, Sequence
from keras.applications.resnet50 import preprocess_input as preprocess_resnet
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import histomicstk.preprocessing.color_normalization as htk_cnorm
import os
import random
from shutil import copyfile

class BaseGenerator(Sequence):
    def __init__(self, config):

        self.config = config

        self.path_mal_train, _, self.files_malignant_train = next(os.walk( os.path.join(self.config['data_path'], 'train', 'malignant', 'rgb') ))
        self.path_ben_train, _, self.files_benign_train = next(os.walk( os.path.join(self.config['data_path'],'train', 'benign', 'rgb') ))
        self.path_mal_cv, _, self.files_malignant_cv = next(os.walk( os.path.join(self.config['data_path'],'val', 'malignant', 'rgb') ))
        self.path_ben_cv, _, self.files_benign_cv = next(os.walk( os.path.join(self.config['data_path'],'val', 'benign', 'rgb') ))
        self.path_mal_test, _, self.files_malignant_test = next(os.walk( os.path.join(self.config['data_path'],'test', 'malignant', 'rgb') ))
        self.path_ben_test, _, self.files_benign_test = next(os.walk( os.path.join(self.config['data_path'],'test', 'benign', 'rgb') ))

        self.batch_size = self.config.trainer.batch_size
        self.label = self.config.label

        self.mal_paths_train = glob.glob( os.path.join(self.path_mal_train, '*') )
        self.ben_paths_train = glob.glob( os.path.join(self.path_ben_train, '*') )
        self.mal_paths_cv = glob.glob( os.path.join(self.path_mal_cv, '*') )
        self.ben_paths_cv = glob.glob( os.path.join(self.path_ben_cv, '*') )
        self.mal_paths_test = glob.glob( os.path.join(self.path_mal_test, '*') )
        self.ben_paths_test = glob.glob( os.path.join(self.path_ben_test, '*') )

        self.mal_outputs_train = [self.label['malignant']] * len(self.mal_paths_train)
        self.ben_outputs_train = [self.label['benign']] * len(self.ben_paths_train)
        self.mal_outputs_cv = [self.label['malignant']] * len(self.mal_paths_cv)
        self.ben_outputs_cv = [self.label['benign']] * len(self.ben_paths_cv)
        self.mal_outputs_test = [self.label['malignant']] * len(self.mal_paths_test)
        self.ben_outputs_test = [self.label['benign']] * len(self.ben_paths_test)


        self.train_paths = self.mal_paths_train + self.ben_paths_train
        self.train_outputs = self.mal_outputs_train + self.ben_outputs_train
        self.cv_paths = self.mal_paths_cv + self.ben_paths_cv
        self.cv_outputs = self.mal_outputs_cv + self.ben_outputs_cv
        self.test_paths = self.mal_paths_cv + self.ben_paths_cv
        self.test_outputs = self.mal_outputs_cv + self.ben_outputs_cv


        z = zip(self.train_paths, self.train_outputs)
        random.shuffle(z)
        self.train_paths, self.train_outputs = zip(*z)




        #self.train_paths = self.train_paths[0 : self.config.len_train]
        #self.train_outputs = self.train_outputs[0 : self.config.len_train]
        #self.cv_paths = self.cv_paths[0 : self.config.len_CV]
        #self.cv_outputs = self.cv_outputs[0 : self.config.len_CV]

        self.ref_std_lab=(0.57506023, 0.10403329, 0.01364062)
        self.ref_mu_lab=(8.63234435, -0.11501964, 0.03868433)

        if os.path.isfile('configs/stats.pkl'):
            with open('configs/stats.pkl', 'rb') as f:
                self.stats = pickle.load(f)
            print 'Stats loaded'
            self.config['stats'] = self.stats
        else:
            print 'No stats file found (To obtain Mu and Sigma from original whole image).'


    def __len__(self):
        raise NotImplementedError


    def __getitem__(self, idx):
        raise NotImplementedError


    def on_epoch_end(self):
        z = zip(self.train_paths, self.train_outputs)
        random.shuffle(z)
        self.train_paths, self.train_outputs = zip(*z)

    def preprocess(self, img_path):

        img = skimage.io.imread(img_path)
        if img.shape == (1024, 1024, 3):
                img = img[::4, ::4, :]

        image_id = int(float(re.findall("\d+\.\d+", img_path)[0]))

        if image_id in self.stats.keys():
            [src_mu, src_sigma] = self.stats[image_id]
            img_nmzd = htk_cnorm.reinhard(img, self.ref_mu_lab, self.ref_std_lab, src_mu=src_mu, src_sigma=src_sigma).astype('float')
        else:
            print '#### stats for %d not present' % (image_id)
            img_nmzd = htk_cnorm.reinhard(img, self.ref_mu_lab, self.ref_std_lab).astype('float')


        img = preprocess_resnet(img_nmzd)

        return img



class RGBTrainGenerator(BaseGenerator):

    def __init__(self, config):
        super(RGBTrainGenerator, self).__init__(config)

        src = self.config.config_path
        dest = os.path.join(self.config.results, 'model.config')
        copyfile(src, dest)
        print 'Copied config file'

        print 'Train Length : ', len(self.train_paths)
        self.datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

        #percent = 0.1
        #self.mal_paths_train = self.mal_paths_train[0 : int(percent * len(self.mal_paths_train))]
        #self.ben_paths_train = self.ben_paths_train[0 : int(percent * len(self.ben_paths_train))]
        #self.mal_outputs_train = self.mal_outputs_train[0 : int(percent * len(self.mal_outputs_train))]
        #self.ben_outputs_train = self.ben_outputs_train[0 : int(percent * len(self.ben_outputs_train))]


        print('# Malignant Samples : %d' % len(self.mal_paths_train))
        print('# Benign    Samples : %d' % len(self.ben_paths_train))

        self.train_paths = self.mal_paths_train + self.ben_paths_train
        self.train_outputs = self.mal_outputs_train + self.ben_outputs_train

        z = zip(self.train_paths, self.train_outputs)
        random.shuffle(z)
        self.train_paths, self.train_outputs = zip(*z)


    def __len__(self):
        return len(self.train_paths) / (self.batch_size)


    def __getitem__(self, idx):

        batchx = self.train_paths[ int(idx*self.batch_size) : int((idx+1)*self.batch_size) ]
        batchy = self.train_outputs[ int(idx*self.batch_size) : int((idx+1)*self.batch_size) ]

        X = np.zeros((self.batch_size, 256, 256, 3))
        Y = np.zeros((self.batch_size, 2))

        for i in range(self.batch_size):
            X[i] = self.preprocess(batchx[i])
            Y[i] = to_categorical(batchy[i], num_classes=2)

        #itr = self.datagen.flow(X, batch_size=self.batch_size)
        #X = itr.next()

        return (X, Y)





class RGBCVGenerator(BaseGenerator):

    def __init__(self, config):
        super(RGBCVGenerator, self).__init__(config)
        print 'CV Length : ', len(self.cv_paths)


    def __len__(self):
        return len(self.cv_paths) / self.batch_size


    def __getitem__(self, idx):
        batchx = self.cv_paths[ idx*self.batch_size : (idx+1)*self.batch_size ]
        batchy = self.cv_outputs[ idx*self.batch_size : (idx+1)*self.batch_size ]

        X = np.zeros((self.batch_size, 256, 256, 3))
        Y = np.zeros((self.batch_size, 2))

        for i in range(self.batch_size):
            X[i] = self.preprocess(batchx[i])
            Y[i] = to_categorical(batchy[i], num_classes=2)

        return (X, Y)





def RGBTestData(config):

    print 'loading RGB data'
    path_mal_test, _, files_malignant_test = next(os.walk( os.path.join(config.test_dir, 'malignant', 'rgb') ))
    path_ben_test, _, files_benign_test = next(os.walk( os.path.join(config.test_dir, 'benign', 'rgb') ))

    mal_paths_test = glob.glob( os.path.join(path_mal_test, '*') )
    ben_paths_test = glob.glob( os.path.join(path_ben_test, '*') )

    mal_outputs_test = [config.label.malignant] * len(mal_paths_test)
    ben_outputs_test = [config.label.benign] * len(ben_paths_test)

    test_paths = mal_paths_test + ben_paths_test
    test_outputs = mal_outputs_test + ben_outputs_test

    z = zip(test_paths, test_outputs)
    random.shuffle(z)
    test_paths, test_outputs = zip(*z)

    ref_std_lab=(0.57506023, 0.10403329, 0.01364062)
    ref_mu_lab=(8.63234435, -0.11501964, 0.03868433)

    if os.path.isfile('configs/stats.pkl'):
        with open('configs/stats.pkl', 'rb') as f:
            stats = pickle.load(f)
        print '###################  Stats loaded Test ####################'
        config['stats'] = stats
    else:
        print 'No stats file found (To obtain Mu and Sigma from original whole image).'

    len_test = len(test_outputs)

    X = np.zeros((len_test, 256, 256, 3))
    Y = [-1] * len_test


    for i in range(len_test):

        #img = skimage.io.imread(test_paths[i])[::4, ::4, :]
        img = skimage.io.imread(test_paths[i])
        if img.shape == (1024, 1024, 3):
                img = img[::4, ::4, :]

        image_id = int(float(re.findall("\d+\.\d+", test_paths[i])[0]))

        if image_id in stats.keys():
            [src_mu, src_sigma] = stats[image_id]
            img_nmzd = htk_cnorm.reinhard(img, ref_mu_lab, ref_std_lab, src_mu=src_mu, src_sigma=src_sigma).astype('float')
        else:
            print '#### stats for %d not present' % (image_id)
            img_nmzd = htk_cnorm.reinhard(img, ref_mu_lab, ref_std_lab).astype('float')

        img = preprocess_resnet(img_nmzd)

        X[i] = img
        Y[i] = test_outputs[i] #to_categorical(test_outputs[i], num_classes=2)

    return (X, Y)
