import numpy as np
import scipy
import matplotlib.pyplot as plt

import dask

from statsmodels.sandbox.distributions.extras import mvnormcdf
from scipy.stats import multivariate_normal

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()


rtda = importr("TDA")
rmph = importr("mph")

# R - GUDHI interface
robjects.r('''
        ComputeDiagramGUDHI <- function(X, max_dimension, max_scale) {
            res <- ripsDiag(X, maxdimension=max_dimension, maxscale=max_scale, library = "GUDHI")
            Diag <- as.matrix(res$diagram)
            return( Diag )
        }
        ''')
ComputeDiagramGUDHI = robjects.globalenv['ComputeDiagramGUDHI']

robjects.r('''
        ComputeLandscapeGUDHI <- function(X, max_dimension, max_scale, k=50, samples=100) {
            res <- ripsDiag(X, maxdimension=max_dimension, maxscale=max_scale, library = "GUDHI")
            Diag <- res$Diagram
            print(colnames(as.matrix(Diag)))
            tseq <-  seq(min(Diag[,2:3]), max(Diag[,2:3]), length=samples)
            lscape <- landscape(Diag, KK=1:k, tseq=tseq)
            return( lscape )
        }
        ''')
ComputeLandscapeGUDHI = robjects.globalenv['ComputeLandscapeGUDHI']

# R - Dionysus interface
robjects.r('''
        ComputeDiagramDionysus <- function(X, max_dimension, max_scale) {
            res <- ripsDiag(X, maxdimension=max_dimension, maxscale=max_scale, library = "Dionysus")
            Diag <- as.matrix(res$diagram)
            return( Diag )
        }
        ''')
ComputeDiagramDionysus = robjects.globalenv['ComputeDiagramDionysus']

robjects.r('''
        ComputeLandscapeDionysus <- function(X, max_dimension, max_scale, k=50, samples=100) {
            res <- ripsDiag(X, maxdimension=max_dimension, maxscale=max_scale, library = "Dionysus")
            Diag <- res$Diagram
            tseq <-  seq(min(Diag[,2:3]), max(Diag[,2:3]), length=samples)
            return( landscape(Diag, KK=1:k, tseq=tseq) )
        }
        ''')
ComputeLandscapeDionysus = robjects.globalenv['ComputeLandscapeDionysus']

# R - MPH interface
robjects.r('''
        ComputeDiagramMPH <- function(X, max_dimension, seed=NULL) {
            if(!is.null(seed)) {
                print('Setting user defined seed for random number generator')
                set.seed(seed)
            }
            gmra <- gmra.create.ikm(X, eps=0, nKids=2^max_dimension, stop=4)
            Diag <- multiscale.rips(gmra, maxD = max_dimension)
            Diag <- Diag$diagram[, c(4, 1, 2)]
            return( Diag )
        }
        ''')
ComputeDiagramMPH = robjects.globalenv['ComputeDiagramMPH']

robjects.r('''
        ComputeLandscapeR <- function(Diag, dimension, scale_range, k, samples) {

            colnames(Diag) <- c("dimension", "Birth", "Death")

            scale_seq <- seq(scale_range[1], scale_range[2], length=samples)

            lscape <- landscape(Diag, dimension=dimension, KK=1:k, tseq=scale_seq)

            res <- list("landscale" = lscape, "scale_seq" = scale_seq)
            return( res )
        }
        ''')
ComputeLandscapeR = robjects.globalenv['ComputeLandscapeR']


def compute_persistence_landscape(diag, dimension, scale_range=None,
                                  k=50, samples=50):

    if scale_range is None:

        bdPairs = diag[diag[:, 0] == dimension, 1:]
        scale_range = np.array([bdPairs.min(), bdPairs.max()])

    res = ComputeLandscapeR(diag, dimension, scale_range, k, samples)

    landscape = np.asarray(res[0]).T
    scale_seq = np.asarray(res[1])

    return landscape, scale_seq


# persistence images
def compute_persistence_image(bd_pairs, dimension, out_res, max_dist,
                              max_persistence, sigma=0.05,
                              boundary_mode='hard'):

    assert(boundary_mode in ['soft', 'hard'])
    assert(dimension in [0, 1])
    
    if dimension == 0:
        out_image = np.zeros((out_res, 1))
    else:
        out_image = np.zeros((out_res, out_res))

    if bd_pairs is None:
        return out_image
    
    def weight_function(persistence, max_persistence):
        w = persistence / np.double(max_persistence)
        if w > 1:
            w = 1
        return w

    bp_pairs = bd_pairs.copy()
    bp_pairs[:, 1] = bp_pairs[:, 1] - bp_pairs[:, 0]

    ubound = max_dist
    if boundary_mode == 'soft':
        ubound += 3 * sigma

    if dimension == 0:

        w = [weight_function(bp_pairs[i, 1], max_persistence)
             for i in range(bp_pairs.shape[0])]

        for pid in range(bp_pairs.shape[0]):

            out_image += w[pid] * _compute_mvnorm_image_dim_1(
                ubound, out_res, bp_pairs[pid, 1], sigma)

    else:

        sigma = np.diag([sigma, sigma])

        bvals = np.linspace(0, ubound, out_res)
        pvals = np.linspace(0, ubound, out_res)

        H, _, _ = np.histogram2d(bp_pairs[:, 0], bp_pairs[:, 1],
                                 bins=(bvals, pvals))
    
        bind, pind = np.nonzero(H)

        for i in range(len(pind)):
            cur_bval = bvals[bind[i]]
            cur_pval = pvals[pind[i]]
            cur_w = weight_function(cur_pval, max_persistence)
            cur_h = H[bind[i]][pind[i]]
            out_image += cur_h * cur_w * _compute_mvnorm_image_dim_2(
                ubound, out_res, [cur_bval, cur_pval], sigma)
        
    return out_image


def _compute_mvnorm_image_dim_2(upper_bound, out_res, mu, sigma):

    out_image = np.zeros([out_res]*2)

    x_vals = np.linspace(0, upper_bound, out_res)
    y_vals = np.linspace(0, upper_bound, out_res)

    X, Y = np.meshgrid(x_vals, y_vals)
    # pos = np.dstack((X, Y))
    pos = zip(X.ravel(), Y.ravel())

    rv = multivariate_normal(mu, sigma)

    out_image = rv.pdf(pos)

    out_image *= (float(upper_bound) / out_res)**2

    out_image = out_image.reshape(out_res, out_res)

    return out_image


def _compute_mvnorm_image_dim_1(upper_bound, out_res, mu, sigma):

    x_vals = np.linspace(0, upper_bound, out_res + 1)
    out_image = scipy.stats.norm.cdf(x_vals, loc=mu, scale=sigma)
    out_image = out_image[1:] - out_image[:-1]

    return out_image.reshape(len(out_image), -1)


# Visualization functions
def plot_birth_persistence_diagram(bd_mat, inf_val=np.inf, fontsize=None, **kwargs):

    lifetime = bd_mat[:, 1] - bd_mat[:, 0]
    inf_life = lifetime >= inf_val
    num_inf = inf_life.sum()

    x = bd_mat[:, 0] # birth
    y = lifetime # death

    if np.isinf(inf_val):
        max_y = y[~inf_life].max()
    else:
        max_y = inf_val

    plt.scatter(x[~inf_life], y[~inf_life], c='b', **kwargs)

    if num_inf > 0:
        plt.scatter(x[inf_life], max_y * np.ones(inf_life.sum()), c='r', **kwargs)

    plt.xlim([0, max_y])
    plt.ylim([0, max_y])
    plt.xlabel('birth', fontsize=fontsize)

    if num_inf > 0:
        plt.ylabel('persistence (%d Inf values)' % num_inf)
    else:
        plt.ylabel('persistence', fontsize=fontsize)

def plot_persistence_diagram(bd_mat, inf_val=np.inf, fontsize=None, **kwargs):

    lifetime = bd_mat[:, 1] - bd_mat[:, 0]
    inf_life = lifetime >= inf_val
    num_inf = inf_life.sum()

    x = bd_mat[:, 0] # birth
    y = bd_mat[:, 1] # death

    if np.isinf(inf_val):
        max_y = y[~inf_life].max()
    else:
        max_y = inf_val

    plt.scatter(x[~inf_life], y[~inf_life], c='b', **kwargs)

    if num_inf > 0:
        plt.scatter(x[inf_life], max_y * np.ones(inf_life.sum()), c='r', **kwargs)

    plt.plot([0, max_y], [0, max_y], 'g-')
    plt.xlim([0, max_y])
    plt.xlabel('birth', fontsize=fontsize)

    if num_inf > 0:
        plt.ylabel('death (%d Inf values)' % num_inf)
    else:
        plt.ylabel('death', fontsize=fontsize)

def plot_birth_death_bars(bd_mat, order='birth', inf_val=np.inf, **kwargs):

    if order == 'birth':
        bd_mat = bd_mat[bd_mat[:, 0].argsort(), ]
    elif order == 'death':
        bd_mat = bd_mat[bd_mat[:, 1].argsort(), ]
    elif order == 'lifetime':
        lifetime = bd_mat[:, 1] - bd_mat[:, 0]
        bd_mat = bd_mat[lifetime.argsort(), ]
    else:
        error("'Invalid value passed for parameter order. Must be either 'birth' or 'death'")

    lifetime = bd_mat[:, 1] - bd_mat[:, 0]
    inf_life = lifetime >= inf_val
    max_life = lifetime[~inf_life].max()

    bd_mat[inf_life, 1] = max_life * 1.25

    for i in np.arange(bd_mat.shape[0]):

        if inf_life[i]:
            c = 'r'
        else:
            c = 'b'

        plt.plot(bd_mat[i, :], [i, i], c + '-', **kwargs)
        plt.plot(bd_mat[i, 0], i, c + 'o', **kwargs)
