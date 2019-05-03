import sys
import json
import glob
import traceback
import numpy as np
import scipy as sp
import pandas as pd
import skimage.io

import os, sys, time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cPickle as pickle

import large_image

import histomicstk as htk
import histomicstk.preprocessing.color_conversion as htk_ccvt
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.filters.shape as htk_shape_filters
import histomicstk.segmentation as htk_seg
import histomicstk.features as htk_features

import tda_utils

from tensorflow.python.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import dask
import dask.distributed
import dask.diagnostics

import warnings
warnings.filterwarnings('ignore')

labelsize = 30
kwargs = {
    "out_res":32,  #224,
    "max_dist":175,
    "sigma_factor":3,
    "nuclei_pts":None,
    "display_result":False
}

label = {'malignant':1, 'benign':0}
patch_size = 1024
overlap = 256
stride = patch_size - overlap
ext_original = '.tif'
ext_patch = '.jpg'

rootDir = sys.argv[1]#'/home/suraj.maniyar/Project/'


annot = np.loadtxt( os.path.join(rootDir, 'data', 'binary_label.txt') ).astype('int')
annot = np.array(annot)
mal_ids = [i for i in range(717) if annot[i,1]==1]
ben_ids = [i for i in range(717) if annot[i,1]==2]
path_mal, dirs, files_malignant = next(os.walk( os.path.join(rootDir, 'data', 'patches', 'malignant') ))
path_ben, dirs, files_benign = next(os.walk( os.path.join(rootDir, 'data', 'patches', 'benign') ))
path_persistence_mal, dirs, files_persistence_mal = next(os.walk( os.path.join(rootDir, 'data', 'persistence_images', 'malignant') ))
path_persistence_ben, dirs, files_persistence_ben = next(os.walk( os.path.join(rootDir, 'data', 'persistence_images', 'benign') ))
path_persistence, dirs, files_persistence = next(os.walk( os.path.join(rootDir, 'data', 'persistence_images') ))


# Returns class (1 for malignant and 0 for benign) for given id_
def get_class(id_):
    # using for loop just in case annotation file is changed
    for i in range(annot.shape[0]):
        if(annot[i,0] == id_):
            if(annot[i, 1] == 1):
                return label['malignant']
            else:
                return label['benign']


# Returns array with dimension 1 : id and dimension 2 : # of patches for that id (In descending order)
def patches_summary_mal():
    path_mal, dirs, files_malignant = next(os.walk( os.path.join(rootDir, 'data', 'patches', 'malignant') ))
    arr = []
    for i in range(len(mal_ids)):
        patches_mal = [files_malignant[j] for j in range(len(files_malignant))
                       if files_malignant[j].startswith(str(mal_ids[i])+'.')]
        arr.append([mal_ids[i], len(patches_mal)])

    return np.array(sorted(arr,key=lambda x: x[1], reverse=True))


# Returns array with dimension 1 : id and dimension 2 : # of patches for that id (In descending order)
def patches_summary_ben():
    path_ben, dirs, files_benign = next(os.walk( os.path.join(rootDir, 'data', 'patches', 'benign') ))
    arr = []
    for i in range(len(ben_ids)):
        patches_ben = [files_benign[j] for j in range(len(files_benign))
                       if files_benign[j].startswith(str(ben_ids[i])+'.')]
        arr.append([ben_ids[i], len(patches_ben)])

    return np.array(sorted(arr,key=lambda x: x[1], reverse=True))


def patches_summary_ben_persistence():
    path_ben, dirs, files_benign = next(os.walk( os.path.join(rootDir, 'data', 'persistence_images', 'benign') ))
    arr = []
    for i in range(len(ben_ids)):
        patches_ben = [files_benign[j] for j in range(len(files_benign))
                       if files_benign[j].startswith(str(ben_ids[i])+'.')]
        arr.append([ben_ids[i], len(patches_ben)])

    return np.array(sorted(arr,key=lambda x: x[1], reverse=True))


def patches_summary_mal_persistence():
    path_mal, dirs, files_malignant = next(os.walk( os.path.join(rootDir, 'data', 'persistence_images', 'malignant') ))
    arr = []
    for i in range(len(mal_ids)):
        patches_mal = [files_malignant[j] for j in range(len(files_malignant))
                       if files_malignant[j].startswith(str(mal_ids[i])+'.')]
        arr.append([mal_ids[i], len(patches_mal)])

    return np.array(sorted(arr,key=lambda x: x[1], reverse=True))


def get_patch_files_mal(id_):
    assert(id_>-1 and id_<717), 'Enter correct id (0-716)'
    path_mal = os.path.join(rootDir, 'data', 'patches', 'malignant')
    patches_mal = glob.glob(os.path.join(path_mal, str(id_) + '.*'))
    patches_mal = [os.path.basename(f) for f in patches_mal]
    return path_mal, patches_mal


def get_patch_files_ben(id_):
    assert(id_>-1 and id_<717), 'Enter correct id (0-716)'
    path_ben = os.path.join(rootDir, 'data', 'patches', 'benign')
    patches_ben = glob.glob(os.path.join(path_ben, str(id_) + '.*'))
    patches_ben = [os.path.basename(f) for f in patches_ben]
    return path_ben, patches_ben


def get_patch_files_mal_persistence(id_):
    assert(id_>-1 and id_<717), 'Enter correct id (0-716)'
    path_mal = os.path.join(rootDir, 'data', 'persistence_images', 'malignant')
    patches_mal = glob.glob(os.path.join(path_mal, str(id_) + '.*'))
    patches_mal = [os.path.basename(f) for f in patches_mal]
    return path_mal, patches_mal

def get_patch_files_ben_persistence(id_):
    assert(id_>-1 and id_<717), 'Enter correct id (0-716)'
    path_ben = os.path.join(rootDir, 'data', 'persistence_images', 'benign')
    patches_ben = glob.glob(os.path.join(path_ben, str(id_) + '.*'))
    patches_ben = [os.path.basename(f) for f in patches_ben]
    return path_ben, patches_ben


# Returns array of patches for a given id
# Return numpy array : (#patches, width, height, n_channels)
def get_patches_from_folder(id_):

    file_loc = rootDir+'data/patches_dump/img_'+str(id_)+'.pkl'

    if os.path.isfile(file_loc):
        f = open(file_loc, 'rb')
        X = pickle.load(f)
        f.close()
        return X

    else:
        typeof = get_class(id_)

        if(typeof == label['malignant']):
            path, patch_names = get_patch_files_mal(id_)
        elif(typeof == label['benign']):
            path, patch_names = get_patch_files_ben(id_)

        X = []
        for i in range(len(patch_names)):
            img = skimage.io.imread(os.path.join(path, patch_names[i]))

            X.append(img)

        return np.array(X)


def detect_nuclei(im_input, min_radius=6, max_radius=10, display_result=False):

    # color normalization
    ref_mu_lab=(8.63234435, -0.11501964, 0.03868433)
    ref_std_lab=(0.57506023, 0.10403329, 0.01364062)

    im_nmzd = htk_cnorm.reinhard(im_input, ref_mu_lab, ref_std_lab)

    # color deconvolution
    w_est = htk_cdeconv.rgb_separate_stains_macenko_pca(im_nmzd, 255)
    nuclear_chid = htk_cdeconv.find_stain_index(htk_cdeconv.stain_color_map['hematoxylin'], w_est)
    im_nuclei_stain = htk_cdeconv.color_deconvolution(im_nmzd, w_est, 255).Stains[:, :, nuclear_chid]

    # segment nuclei foreground
    th = skimage.filters.threshold_li(im_nuclei_stain) * 0.8
    # th = skimage.filters.threshold_otsu(im_nuclei_stain)
    im_fgnd_mask = im_nuclei_stain < th
    im_fgnd_mask = skimage.morphology.opening(im_fgnd_mask, skimage.morphology.disk(2))
    im_fgnd_mask = skimage.morphology.closing(im_fgnd_mask, skimage.morphology.disk(1))

    # detect nuclei
    im_dog, im_dog_sigma = htk_shape_filters.cdog(im_nuclei_stain, im_fgnd_mask,
                                                  sigma_min=min_radius / np.sqrt(2),
                                                  sigma_max=max_radius / np.sqrt(2))

    nuclei_coord = skimage.feature.peak_local_max(im_dog, min_distance=min_radius/2, threshold_rel=0.1)

    nuclei_coord = nuclei_coord[im_fgnd_mask[nuclei_coord[:, 0], nuclei_coord[:, 1]], :]

    nuclei_rad = np.array([im_dog_sigma[nuclei_coord[i, 0], nuclei_coord[i, 1]] * np.sqrt(2)
                           for i in range(nuclei_coord.shape[0])])

    # display result
    if display_result:

        print 'Number of nuclei = ', nuclei_coord.shape[0]

        plt.figure(figsize=(30,20))
        plt.subplot(2, 2, 1)
        plt.imshow(im_input)
        plt.title('Input', fontsize=labelsize)
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(im_nuclei_stain)
        plt.title('Deconv nuclei stain', fontsize=labelsize)
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(im_fgnd_mask)
        plt.title('Foreground mask', fontsize=labelsize)
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(im_nmzd)
        plt.plot(nuclei_coord[:, 1], nuclei_coord[:, 0], 'k+')

        for i in range(nuclei_coord.shape[0]):

            cx = nuclei_coord[i, 1]
            cy = nuclei_coord[i, 0]
            r = nuclei_rad[i]

            mcircle = mpatches.Circle((cx, cy), r, color='g', fill=False)
            plt.gca().add_patch(mcircle)

        plt.title('Nuclei detection', fontsize=labelsize)
        plt.axis('off')

        plt.tight_layout()

    return nuclei_coord, nuclei_rad


def compute_nuclei_persistence_diagram(im_input, inf_val=175, nuclei_pts=None,
                                       display_result=False):

    # Detect nuclei centroids
    if nuclei_pts is None:

        tic = time.time()
        nuclei_pts, nuclei_rad = detect_nuclei(im_input, display_result=False)
        toc = time.time()
        print 'Nuclei detection: %d nuclei, %.2f seconds' % (len(nuclei_rad), (toc - tic))

    # Compute persistence diagram
    tic = time.time()

    dgm_mph = np.asarray(tda_utils.ComputeDiagramMPH(nuclei_pts, 1))
    bd_pairs_mph = [dgm_mph[dgm_mph[:, 0] == i, 1:] for i in range(2)]
    

    toc = time.time()
    print 'Persistence diagram computation: Dim0 - %d points, Dim1 - %d points, %.2f seconds' % (
        bd_pairs_mph[0].shape[0], bd_pairs_mph[1].shape[0], (toc - tic))

    # display result
    if display_result:

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(im_input)
        plt.title('Input')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(im_input)
        plt.plot(nuclei_pts[:, 1], nuclei_pts[:, 0], 'g+')

        plt.title('Nuclei detection')
        plt.axis('off')

        plt.figure(figsize=(14, 7))
        for i in range(2):

            plt.subplot(1, 2, i+1)
            tda_utils.plot_persistence_diagram(bd_pairs_mph[i], inf_val=inf_val)
            plt.title('Persistence diagram (dim=%d, #points=%d)' % (i, bd_pairs_mph[i].shape[0]))

        plt.tight_layout()

    return bd_pairs_mph, dgm_mph


def compute_nuclei_persistence_image(im_input, out_res=224, max_dist=175,
                                     sigma_factor=8, nuclei_pts=None,
                                     display_result=False):

    # Detect nuclei centroids
    if nuclei_pts is None:

        tic = time.time()
        nuclei_pts, nuclei_rad = detect_nuclei(im_input, display_result=False)
        toc = time.time()
        print 'Nuclei detection: %d nuclei, %.2f seconds' % (len(nuclei_rad), (toc - tic))

    # compute nuclei persistence diagram
    bd_pairs_mph, dgm_mph = compute_nuclei_persistence_diagram(im_input, inf_val=max_dist,
                                                               nuclei_pts=nuclei_pts)

    # Compute persistence image
    sigma = sigma_factor * max_dist / out_res

    tic = time.time()

    # im_pi_dim_0 = tda_utils.compute_persistence_image(
    #    bd_pairs_mph[0], 0, out_res, max_dist, max_dist, sigma=sigma)
    
    im_pi_dim_1 = tda_utils.compute_persistence_image(
        bd_pairs_mph[1], 1, out_res, max_dist, max_dist, sigma=sigma)

    toc = time.time()

    print 'Persistence image computation: %.2f seconds' % (toc - tic)

    # display result
    if display_result:

        plt.figure(figsize=(16, 8))

        plt.subplot(1, 2, 1)
        plt.imshow(im_input)
        plt.title('Input')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(im_input)
        plt.plot(nuclei_pts[:, 1], nuclei_pts[:, 0], 'g+')

        plt.title('Nuclei detection')
        plt.axis('off')

        plt.figure(figsize=(16, 8))
        for i in range(2):

            plt.subplot(1, 2, i+1)
            tda_utils.plot_birth_persistence_diagram(bd_pairs_mph[i], inf_val=max_dist)
            plt.title('Birth-persistence diagram (dim=%d, #points=%d)' % (i, bd_pairs_mph[i].shape[0]))

        plt.figure(figsize=(16, 8))

        plt.subplot(1, 2, 1)
        x_vals = np.linspace(0, max_dist, out_res+1)[:-1]
        plt.plot(x_vals, im_pi_dim_0)
        plt.xlabel('Birth')
        plt.ylabel('Persistence')

        plt.subplot(1, 2, 2)
        plt.imshow(im_pi_dim_1, cmap=plt.cm.hot, origin='lower', extent=[0, max_dist, 0, max_dist])
        plt.xlabel('Birth')
        plt.ylabel('Persistence')

        plt.tight_layout()

    # return im_pi_dim_0, im_pi_dim_1
    return im_pi_dim_1, bd_pairs_mph, dgm_mph


def get_persistence_images(id_):
    typeof = get_class(id_)
    X = []

    if typeof == label['malignant']:

        id_files = [files_persistence_mal[i]
                    for i in range(len(files_persistence_mal))
                    if files_persistence_mal[i].startswith(str(id_)+'.')]
        for file_name in id_files:
            # print file_name
            file_loc = os.path.join( path_persistence_mal, file_name )

            f = open(file_loc, 'rb')
            # Converting single channel image to 3 channels by copying the same image
            arr = np.array( [pickle.load(f)]*3 )
            arr = np.moveaxis(arr, 0, 2)
            if(arr.shape == (224, 224, 3)):
                X.append(arr)
            f.close()

    elif typeof == label['benign']:

        id_files = [files_persistence_ben[i]
                    for i in range(len(files_persistence_ben))
                    if files_persistence_ben[i].startswith(str(id_)+'.')]

        for file_name in id_files:
            # print file_name
            file_loc = os.path.join( path_persistence_ben, file_name )

            f = open(file_loc, 'rb')
            # f = open(file_loc, 'rb')
            # Converting single channel image to 3 channels by copying the same image
            arr = np.array( [pickle.load(f)]*3 )
            arr = np.moveaxis(arr, 0, 2)

            X.append( arr )
            f.close()

    return np.array(X)

# If persistence is True, function returns persistence data : (Length, 224, 224, 3)
# else if persistence is set False, function returns RGB data : (Length, 1024, 1024, 3)
# pass preprocess_func as argument the function which you want to run on a single image.

def get_data(train_ids, cv_ids, test_ids, persistence=False, preprocess_func=None):

    print 'Loading data'

    #global len_train, len_CV, len_test, batches_train, batches_CV, batches_test

    X_train, Y_train = [], []
    X_CV, Y_CV = [], []
    X_test, Y_test = [], []

    TIC = time.time()


    # Train
    print '-'*100
    for i in range(len(train_ids)):
        tic = time.time()
        if persistence:
            X=  get_persistence_images(train_ids[i])
        else:
            X = get_patches_from_folder(train_ids[i])
        Y = get_class(train_ids[i])
        X_train.append(X)
        for j in range(X.shape[0]):
            Y_train.append(Y)

        print('\t %d / %d  : \t id = %d \t #Patches = %d \t time = %f secs' % (i, len(train_ids), train_ids[i], X.shape[0], time.time()-tic))

    # Generate numpy arrays and convert class to one hot vector
    X_train = np.vstack(X_train)
    Y_train = to_categorical(Y_train)

    # Shuffling the data
    X_train, _, Y_train, _ = train_test_split(X_train, Y_train, test_size=0, random_state=7)

    if preprocess_func is not None:
        for i in range(X_train.shape[0]):
            X_train[i] = preprocess_func(X_train[i])

    # CV
    print '-'*100
    for i in range(len(cv_ids)):
        tic = time.time()
        if persistence:
            X = get_persistence_images(cv_ids[i])
        else:
            X = get_patches_from_folder(cv_ids[i])
        Y = get_class(cv_ids[i])
        X_CV.append(X)
        for j in range(X.shape[0]):
            Y_CV.append(Y)

        print('\t %d / %d  : \t id = %d \t #Patches = %d \t time = %f secs' % (i, len(cv_ids), cv_ids[i], X.shape[0], time.time()-tic))

    X_CV = np.vstack(X_CV)
    Y_CV = to_categorical(Y_CV)

    if preprocess_func is not None:
        for i in range(X_CV.shape[0]):
            X_CV[i] = preprocess_func(X_CV[i])

    # Test
    print '-'*100
    for i in range(len(test_ids)):
        tic = time.time()
        if persistence:
            X = get_persistence_images(test_ids[i])
        else:
            X = get_patches_from_folder(test_ids[i])
        Y = get_class(test_ids[i])
        X_test.append(X)
        for j in range(X.shape[0]):
            Y_test.append(Y)

        print('\t %d / %d  : \t id = %d \t #Patches = %d \t time = %f secs' % (i, len(test_ids), test_ids[i], X.shape[0], time.time()-tic))

    print '-'*100

    print('Total Time = %f' % (time.time()-TIC))
    print '-'*100

    X_test = np.vstack(X_test)
    Y_test = to_categorical(Y_test)

    if preprocess_func is not None:
        for i in range(X_test.shape[0]):
            X_test[i] = preprocess_func(X_test[i])

    print 'Train:'
    print(X_train.shape)
    print(Y_train.shape)
    print '-'*100

    print 'CV:'
    print(X_CV.shape)
    print(Y_CV.shape)
    print '-'*100

    print 'Test:'
    print(X_test.shape)
    print(Y_test.shape)
    print '-'*100

    return (X_train, Y_train, X_CV, Y_CV, X_test, Y_test)


def get_persistence_image(img_path, **kwargs):
    print '\nComputing for %s\n' % img_path[-15:].replace('/', '').replace('t','').replace('n','')
    img = skimage.io.imread(img_path)
    persistence_img, bd_pairs_mph, dgm_mph = compute_nuclei_persistence_image(
        im_input=img, **kwargs)
    print(persistence_img.shape)
    return persistence_img, bd_pairs_mph, dgm_mph


def save_patch_persistence(img_path, out_pi_file, out_pd_file, **kwargs):

    try:
        
        persistence_img, bd_pairs_mph, dgm_mph = get_persistence_image(
            img_path, **kwargs)

        with open(out_pi_file, 'wb') as f:
            pickle.dump(persistence_img, f)

            persistence_diagram = [bd_pairs_mph, dgm_mph]
            with open(out_pd_file, 'wb') as f:
                pickle.dump(persistence_diagram, f)

                print('\nSaved %s\n\n' % img_path)
               
        return None

    except:
        return img_path + '\n' + str(traceback.format_exc())
        

def save_case_persistence(id_, **kwargs):

    tic = time.time()
    typeof = get_class(id_)
    bad_patches = []
    
    if typeof == label['malignant']:
        
        path_mal, patches_mal = get_patch_files_mal(id_)
        res = []
        
        for i in range(len(patches_mal)):
            
            img_path = os.path.join(path_mal, patches_mal[i])

            file_name = patches_mal[i].replace('.jpg', '.pkl')

            file_path_mal = os.path.join(path_persistence_mal, file_name)
            file_path_pd = os.path.join(path_persistence, 'malignant_pd', file_name)

            if os.path.isfile(file_path_mal) and os.path.isfile(file_path_pd):
                print 'File %s already exists' % file_name
                continue

            res.append(
                dask.delayed(save_patch_persistence)(
                    img_path, file_path_mal, file_path_pd, **kwargs)
            )

        res = dask.compute(*res)

        bad_patches = [res[i]
                       for i in range(len(res))
                       if res[i] is not None]
        
    elif typeof == label['benign']:
        
        path_ben, patches_ben = get_patch_files_ben(id_)

        res = []
        
        for i in range(len(patches_ben)):
            img_path = os.path.join(path_ben, patches_ben[i])

            file_name = patches_ben[i].replace('.jpg', '.pkl')

            file_path_ben = os.path.join(path_persistence_ben, file_name)
            file_path_pd = os.path.join(path_persistence, 'benign_pd', file_name)

            if os.path.isfile(file_path_ben) and os.path.isfile(file_path_pd):
                print 'File %s already exists' % file_name
                continue

            res.append(
                dask.delayed(save_patch_persistence)(
                    img_path, file_path_ben, file_path_pd, **kwargs)
            )

        res = dask.compute(*res)

        bad_patches = [res[i]
                       for i in range(len(res))
                       if res[i] is not None]
            
    print "Time for %d = %.3f secs" % (id_, time.time()-tic)
    print '\n' + '='*100 + '\n'
    return bad_patches


def is_processed(id_):

    # print(id_)
    
    typeof = get_class(id_)

    if typeof == label['malignant']:

        path_mal, patches_mal = get_patch_files_mal(id_)
        for i in range(len(patches_mal)):
            file_name = patches_mal[i].replace('.jpg', '.pkl')

            file_path_mal = os.path.join(path_persistence_mal, file_name)
            file_path_pd = os.path.join(path_persistence, 'malignant_pd', file_name)

            if os.path.isfile(file_path_mal) and os.path.isfile(file_path_pd):
                continue
            else:
                return False
        return True

    elif typeof == label['benign']:

        path_ben, patches_ben = get_patch_files_ben(id_)
        for i in range(len(patches_ben)):

            file_name = patches_ben[i].replace('.jpg', '.pkl')

            file_path_ben = os.path.join(path_persistence_ben, file_name)
            file_path_pd = os.path.join(path_persistence, 'benign_pd', file_name)

            if os.path.isfile(file_path_ben) and os.path.isfile(file_path_pd):
                continue
            else:
                return False

            
def get_cases_to_process(id_list):

    new_id_list = [id_ for id_ in id_list if not is_processed(id_)]
    return new_id_list


def save_persistence_parallelize(id_list, **kwargs):

    id_list = get_cases_to_process(id_list)
    print(id_list)
    
    result = []
    tic = time.time()
    for i in range(len(id_list)):
        result.append(dask.delayed(save_case_persistence)(
            id_list[i], **kwargs))
    result = dask.compute(*result)

    num_bad_patches = 0
    
    f = open('bad_patches.log', 'w')
    for i in range(len(result)):
        if len(result[i]) == 0:
            continue
        num_bad_patches += len(result[i])
        for j in range(len(result[i])):
            f.write(result[i][j] + '\n')
    f.close()

    print('Number of bad patches = {}'.format(num_bad_patches))
    print('Time taken = %d secs' % (time.time() - tic))


if __name__ == "__main__":

    # save_persistence(10)

    bad_cases = []
    with open(os.path.join(rootDir, 'data', 'bad_cases.json')) as f:
        bad_cases = json.load(f)
    
    id_list = [i for i in range(717) if i not in bad_cases]

    # c = dask.distributed.Client('localhost:8786') #'localhost:8000')
    scheduler = dask.distributed.LocalCluster(n_workers=14, threads_per_worker=1)
    c = dask.distributed.Client(scheduler)
    print(c)
    save_persistence_parallelize(id_list, **kwargs)



