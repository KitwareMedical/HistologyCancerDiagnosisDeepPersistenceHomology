import os
import numpy as np
import cPickle as pickle

class BaseDataLoader(object):

    def __init__(self, config):
        self.config = config
        self.label = {'malignant':1, 'benign':0}

        self.dataDir = config['data_path']
        annot_path = os.path.join(self.dataDir, 'binary_label.txt')
        self.annot = np.loadtxt(annot_path).astype('int')

        self.path_mal, _, self.files_malignant = next(os.walk( os.path.join(self.dataDir, 'patches', 'malignant') ))
        self.path_ben, _, self.files_benign = next(os.walk( os.path.join(self.dataDir, 'patches', 'benign') ))
        self.path_persistence_mal, _, self.files_persistence_mal = next(os.walk( os.path.join(self.dataDir, 'persistence_images', 'malignant') ))
        self.path_persistence_ben, _, self.files_persistence_ben = next(os.walk( os.path.join(self.dataDir, 'persistence_images', 'benign') ))
        self.path_persistence, _, self.files_persistence = next(os.walk( os.path.join(self.dataDir, 'persistence_images') ))

        # Separate lists to store malignant and benign indices
        self.mal_ids = [i for i in range(717) if self.annot[i,1]==1]
        self.ben_ids = [i for i in range(717) if self.annot[i,1]==2]

        self.ref_mu_lab=(8.63234435, -0.11501964, 0.03868433)
        self.ref_std_lab=(0.57506023, 0.10403329, 0.01364062)


        if os.path.isfile('configs/stats.pkl'):
            with open('configs/stats.pkl', 'rb') as f:
                self.stats = pickle.load(f)
            print 'Stats loaded'
            self.config['stats'] = self.stats
        else:
            print 'No stats file found (To obtain Mu and Sigma from original whole image).'
    # Returns class (1 for malignant and 0 for benign) for given index
    def get_class(self,index):

        assert(index>-1 and index<717), 'Enter correct index (0-716)'
        if index in self.mal_ids:
            return self.label['malignant']
        else:
            return self.label['benign']




    def get_train_data(self):
        raise NotImplementedError

    def get_cv_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError
