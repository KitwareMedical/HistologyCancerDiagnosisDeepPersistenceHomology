import warnings
warnings.filterwarnings("ignore")

import numpy as np
import skimage.io
import os, sys, time
import re
import matplotlib.pyplot as plt
import cPickle as pickle
import h5py
import glob
from keras.utils import to_categorical, Sequence
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.applications.resnet50 import preprocess_input as preprocess_resnet
import histomicstk.preprocessing.color_normalization as htk_cnorm
import os
import random
from shutil import copyfile


class BaseGenerator(Sequence):
    def __init__(self, config):
        self.config = config
        self.dataDir = config['data_path']

        self.path_mal_train, _, self.files_malignant_train = next(os.walk( os.path.join(self.config['data_path'], 'train', 'malignant', 'rgb') ))
        self.path_ben_train, _, self.files_benign_train = next(os.walk( os.path.join(self.config['data_path'], 'train', 'benign', 'rgb') ))
        self.path_mal_cv, _, self.files_malignant_cv = next(os.walk( os.path.join(self.config['data_path'], 'val', 'malignant', 'rgb') ))
        self.path_ben_cv, _, self.files_benign_cv = next(os.walk( os.path.join(self.config['data_path'], 'val', 'benign', 'rgb') ))

        self.path_mal_train_per, _, _ = next(os.walk( os.path.join(self.config['data_path'], 'train', 'malignant', 'persistence_images') ))
        self.path_ben_train_per, _, _ = next(os.walk( os.path.join(self.config['data_path'], 'train', 'benign', 'persistence_images') ))
        self.path_mal_cv_per, _, _ = next(os.walk( os.path.join(self.config['data_path'], 'val', 'malignant', 'persistence_images') ))
        self.path_ben_cv_per, _, _ = next(os.walk( os.path.join(self.config['data_path'], 'val', 'benign', 'persistence_images') ))

        self.batch_size = self.config.trainer.batch_size
        self.label = self.config.label

        self.mal_paths_train = glob.glob( os.path.join(self.path_mal_train, '*') )
        self.ben_paths_train = glob.glob( os.path.join(self.path_ben_train, '*') )
        self.mal_paths_cv_per = glob.glob( os.path.join(self.path_mal_cv_per, '*') )
        self.ben_paths_cv_per = glob.glob( os.path.join(self.path_ben_cv_per, '*') )

        self.mal_outputs_train = [self.label['malignant']] * len(self.mal_paths_train)
        self.ben_outputs_train = [self.label['benign']] * len(self.ben_paths_train)
        self.mal_outputs_cv_per = [self.label['malignant']] * len(self.mal_paths_cv_per)
        self.ben_outputs_cv_per = [self.label['benign']] * len(self.ben_paths_cv_per)

        self.class_weights = self.config.class_weights

        self.train_paths = self.mal_paths_train + self.ben_paths_train
        self.train_outputs = self.mal_outputs_train + self.ben_outputs_train
        self.cv_paths = self.mal_paths_cv_per + self.ben_paths_cv_per
        self.cv_outputs = self.mal_outputs_cv_per + self.ben_outputs_cv_per

        z = zip(self.train_paths, self.train_outputs)
        random.shuffle(z)
        self.train_paths, self.train_outputs = zip(*z)

        z = zip(self.cv_paths, self.cv_outputs)
        random.shuffle(z)
        self.cv_paths, self.cv_outputs = zip(*z)

        #self.train_paths = self.train_paths[ 0 : self.config.len_train]
        #self.train_outputs = self.train_outputs[ 0 : self.config.len_train]
        #self.cv_paths = self.cv_paths[ 0 : self.config.len_CV]
        #self.cv_outputs = self.cv_outputs[ 0 : self.config.len_CV]

        print('# Malignant Samples : %d' % len(self.mal_paths_train))
        print('# Benign    Samples : %d' % len(self.ben_paths_train))
        
        self.train_files = [os.path.basename(elem) for elem in self.train_paths]
        self.cv_files = [os.path.basename(elem) for elem in self.cv_paths]

        self.train_files = [elem.replace('.jpg', '') for elem in self.train_files]
        self.cv_files = [elem.replace('.pkl', '') for elem in self.cv_files]


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
        z = zip(self.train_files, self.train_outputs)
        random.shuffle(z)
        self.train_files, self.train_outputs = zip(*z)

    def preprocess_persistence(self, img):
        img = img / self.config.trainer.percentile_factor
        img = np.array([img])
        img = np.moveaxis(img, 0, 2)
        return img



class CombinedTrainGenerator(BaseGenerator):

    def __init__(self, config):
        super(CombinedTrainGenerator, self).__init__(config)
        print 'Train Length : ', len(self.train_paths)
        print 'Class Weights = ', self.config.class_weights
        self.datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

        src = self.config.config_path
        dest = os.path.join(self.config.results, 'model.config')
        copyfile(src, dest)
        print 'Copied config file'



    def __len__(self):
        return len(self.train_paths) / self.batch_size


    def __getitem__(self, idx):

        batchx = self.train_files[ idx*self.batch_size : (idx+1)*self.batch_size ]
        batchy = self.train_outputs[ idx*self.batch_size : (idx+1)*self.batch_size ]

        X_RGB = np.zeros((self.batch_size, 256, 256, 3))
        X_Per = np.zeros((self.batch_size, 32, 32, 1))
        Y = np.zeros((self.batch_size, 2))

        for i in range(self.batch_size):
            image_id =  int(float(re.findall("\d+\.\d+", batchx[i])[0]))

            fnameRGB = batchx[i] + '.jpg'
            fnamePer = batchx[i] + '.pkl'

            if batchy[i] == self.label['malignant'] : #image_id in self.mal_ids:
                pathRGB = os.path.join(self.path_mal_train, fnameRGB)
                pathPer = os.path.join(self.path_mal_train_per, fnamePer)

            elif batchy[i] == self.label['benign'] :
                pathRGB = os.path.join(self.path_ben_train, fnameRGB)
                pathPer = os.path.join(self.path_ben_train_per, fnamePer)


            img = skimage.io.imread(pathRGB)
            if img.shape == (1024, 1024, 3):
                img = img[::4, ::4, :]

            image_id = int(float(re.findall("\d+\.\d+", pathRGB)[0]))

            if image_id in self.config['stats'].keys():
                [src_mu, src_sigma] = self.stats[image_id]
                img_nmzd = htk_cnorm.reinhard(img, self.ref_mu_lab, self.ref_std_lab, src_mu=src_mu, src_sigma=src_sigma).astype('float')
            else:
                print '#### stats for %d not present' % (image_id)
                img_nmzd = htk_cnorm.reinhard(img, self.ref_mu_lab, self.ref_std_lab).astype('float')

            imgRGB = preprocess_resnet(img_nmzd)


            with open(pathPer, 'rb') as f:
                img = pickle.load(f)

            imgPer = self.preprocess_persistence(img)

            X_RGB[i] = imgRGB
            X_Per[i] = imgPer
            Y[i] = to_categorical(batchy[i], num_classes=2)

        #itr = self.datagen.flow(X_RGB, batch_size=self.batch_size)
        #X_RGB = itr.next()

        return ([X_RGB, X_Per], Y)





class CombinedCVGenerator(BaseGenerator):

    def __init__(self, config):
        super(CombinedCVGenerator, self).__init__(config)
        print 'CV Length : ', len(self.cv_paths)

    def __len__(self):
        return len(self.cv_paths) / self.batch_size

    def __getitem__(self, idx):

        batchx = self.cv_files[ idx*self.batch_size : (idx+1)*self.batch_size ]
        batchy = self.cv_outputs[ idx*self.batch_size : (idx+1)*self.batch_size ]

        X_RGB = np.zeros((self.batch_size, 256, 256, 3))
        X_Per = np.zeros((self.batch_size, 32, 32, 1))
        Y = np.zeros((self.batch_size, 2))

        for i in range(self.batch_size):
            image_id = int(float(re.findall("\d+\.\d+", batchx[i])[0]))
            fnameRGB = batchx[i] + '.jpg'
            fnamePer = batchx[i] + '.pkl'


            if batchy[i] == self.label['malignant'] : #image_id in self.mal_ids:
                pathRGB = os.path.join(self.path_mal_cv, fnameRGB)
                pathPer = os.path.join(self.path_mal_cv_per, fnamePer)

            elif batchy[i] == self.label['benign'] :
                pathRGB = os.path.join(self.path_ben_cv, fnameRGB)
                pathPer = os.path.join(self.path_ben_cv_per, fnamePer)

            #img = skimage.io.imread(pathRGB)[::4, ::4, :]
            img = skimage.io.imread(pathRGB)
            if img.shape == (1024, 1024, 3):
                img = img[::4, ::4, :]

            image_id = int(float(re.findall("\d+\.\d+", pathRGB)[0]))

            if image_id in self.stats.keys():
                [src_mu, src_sigma] = self.stats[image_id]
                img_nmzd = htk_cnorm.reinhard(img, self.ref_mu_lab, self.ref_std_lab, src_mu=src_mu, src_sigma=src_sigma).astype('float')
            else:
                print '#### stats for %d not present' % (image_ids[i])
                img_nmzd = htk_cnorm.reinhard(img, self.ref_mu_lab, self.ref_std_lab).astype('float')

            imgRGB = preprocess_resnet(img_nmzd)

            with open(pathPer, 'rb') as f:
                img = pickle.load(f)

            imgPer = self.preprocess_persistence(img)

            X_RGB[i] = imgRGB
            X_Per[i] = imgPer
            Y[i] = to_categorical(batchy[i], num_classes=2)


        return ([X_RGB, X_Per], Y)




def CombinedTestData(config):

    print 'Loading combined data'
    path_mal_test, _, files_malignant_test = next(os.walk( os.path.join(config.test_dir, 'malignant', 'rgb') ))
    path_ben_test, _, files_benign_test = next(os.walk( os.path.join(config.test_dir, 'benign', 'rgb') ))

    path_mal_test_per, _, _ = next(os.walk( os.path.join(config.test_dir, 'malignant', 'persistence_images') ))
    path_ben_test_per, _, _ = next(os.walk( os.path.join(config.test_dir, 'benign', 'persistence_images') ))

    batch_size = config.trainer.batch_size
    label = config.label

    mal_paths_test = glob.glob( os.path.join(path_mal_test_per, '*') )
    ben_paths_test = glob.glob( os.path.join(path_ben_test_per, '*') )

    mal_outputs_test = [label['malignant']] * len(mal_paths_test)
    ben_outputs_test = [label['benign']] * len(ben_paths_test)


    test_paths = mal_paths_test + ben_paths_test
    test_outputs = mal_outputs_test + ben_outputs_test

    test_files = [os.path.basename(elem) for elem in test_paths]
    test_files = [elem.replace('.pkl', '') for elem in test_files]


    ref_std_lab=(0.57506023, 0.10403329, 0.01364062)
    ref_mu_lab=(8.63234435, -0.11501964, 0.03868433)

    if os.path.isfile('configs/stats.pkl'):
        with open('configs/stats.pkl', 'rb') as f:
            stats = pickle.load(f)
        print 'Stats loaded'
        config['stats'] = stats
    else:
        print 'No stats file found (To obtain Mu and Sigma from original whole image).'

    len_test = len(test_outputs) #config.len_test#

    X_RGB = np.zeros((len_test, 256, 256, 3))
    X_Per = np.zeros((len_test, 32, 32, 1))
    Y = [-1] * len_test



    for i in range(len_test):

        image_id =  int(float(re.findall("\d+\.\d+", test_files[i])[0]))

        fnameRGB = test_files[i] + '.jpg'
        fnamePer = test_files[i] + '.pkl'

        if test_outputs[i] == config.label['malignant']:
            pathRGB = os.path.join(path_mal_test, fnameRGB)
            pathPer = os.path.join(path_mal_test_per, fnamePer)

        elif test_outputs[i] == config.label['benign']:
            pathRGB = os.path.join(path_ben_test, fnameRGB)
            pathPer = os.path.join(path_ben_test_per, fnamePer)


        #img = skimage.io.imread(pathRGB)[::4, ::4, :]
        img = skimage.io.imread(pathRGB)
        if img.shape == (1024, 1024, 3):
                img = img[::4, ::4, :]

        image_id = int(float(re.findall("\d+\.\d+", pathRGB)[0]))

        if image_id in stats.keys():
            [src_mu, src_sigma] = stats[image_id]
            img_nmzd = htk_cnorm.reinhard(img, ref_mu_lab, ref_std_lab, src_mu=src_mu, src_sigma=src_sigma).astype('float')
        else:
            print '#### stats for %d not present' % (image_id)
            img_nmzd = htk_cnorm.reinhard(img, ref_mu_lab, ref_std_lab).astype('float')

        imgRGB = preprocess_resnet(img_nmzd)

        with open(pathPer, 'rb') as f:
            img = pickle.load(f)
        img = img / config.trainer.percentile_factor
        img = np.array([img])
        imgPer = np.moveaxis(img, 0, 2)

        X_RGB[i] = imgRGB
        X_Per[i] = imgPer
        Y[i] = test_outputs[i]

    print 'RGB : ', X_RGB.shape
    print 'Per : ', X_Per.shape
    print 'len(Y) : ', len(Y)

    return [X_RGB, X_Per, Y]




'''
class CombinedTestGenerator(Sequence):
    def __init__(self, config):
        self.config = config
        self.dataDir = config['data_path']

        self.path_mal_test, _, self.files_malignant_test = next(os.walk( os.path.join(config.test_dir, 'malignant', 'rgb') ))
        self.path_ben_test, _, self.files_benign_test = next(os.walk( os.path.join(config.test_dir, 'benign', 'rgb') ))

        self.path_mal_test_per, _, _ = next(os.walk( os.path.join(config.test_dir, 'malignant', 'persistence_images') ))
        self.path_ben_test_per, _, _ = next(os.walk( os.path.join(config.test_dir, 'benign', 'persistence_images') ))

        self.batch_size = self.config.trainer.batch_size
        self.label = self.config.label

        self.mal_paths_test = glob.glob( os.path.join(self.path_mal_test, '*') )
        self.ben_paths_test = glob.glob( os.path.join(self.path_ben_test, '*') )

        self.mal_outputs_test = [self.label['malignant']] * len(self.mal_paths_test)
        self.ben_outputs_test = [self.label['benign']] * len(self.ben_paths_test)

        self.class_weights = self.config.class_weights

        self.test_paths = self.mal_paths_test + self.ben_paths_test
        self.test_outputs = self.mal_outputs_test + self.ben_outputs_test

        self.test_files = [os.path.basename(elem) for elem in self.test_paths]
        self.test_files = [elem.replace('.jpg', '') for elem in self.test_files]

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
        return len(self.test_paths) / self.batch_size


    def __getitem__(self, idx):

        print '__getitem__ : ', idx
        batchx = self.test_files[ idx*self.batch_size : (idx+1)*self.batch_size ]
        batchy = self.test_outputs[ idx*self.batch_size : (idx+1)*self.batch_size ]

        X_RGB = np.zeros((self.batch_size, 256, 256, 3))
        X_Per = np.zeros((self.batch_size, 224, 224, 3))
        Y = np.zeros((self.batch_size, 2))

        for i in range(self.batch_size):

            image_id =  int(float(re.findall("\d+\.\d+", batchx[i])[0]))

            fnameRGB = batchx[i] + '.jpg'
            fnamePer = batchx[i] + '.pkl'

            if batchy[i] == self.label['malignant']:
                pathRGB = os.path.join(self.path_mal_test, fnameRGB)
                pathPer = os.path.join(self.path_mal_test_per, fnamePer)

            elif batchy[i] == self.label['benign']:
                pathRGB = os.path.join(self.path_ben_test, fnameRGB)
                pathPer = os.path.join(self.path_ben_test_per, fnamePer)


            img = skimage.io.imread(pathRGB)[::4, ::4, :]
            image_id = int(float(re.findall("\d+\.\d+", pathRGB)[0]))

            if image_id in self.stats.keys():
                [src_mu, src_sigma] = self.stats[image_id]
                img_nmzd = htk_cnorm.reinhard(img, self.ref_mu_lab, self.ref_std_lab, src_mu=src_mu, src_sigma=src_sigma).astype('float')
            else:
                print '#### stats for %d not present' % (image_id)
                img_nmzd = htk_cnorm.reinhard(img, self.ref_mu_lab, self.ref_std_lab).astype('float')

            imgRGB = preprocess_resnet(img_nmzd)

            with open(pathPer, 'rb') as f:
                img = pickle.load(f)
            img = np.array([img]*3)
            imgPer = np.moveaxis(img, 0, 2)

            X_RGB[i] = imgRGB
            X_Per[i] = imgPer
            Y[i] = to_categorical(self.test_outputs[idx], num_classes=2)


        return ([X_RGB, X_Per], Y)
'''
