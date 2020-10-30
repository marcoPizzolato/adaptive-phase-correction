import multiprocessing as mp
from phase_correction import complex_adaptive_regularization as cmpxadptvreg
from phase_correction import correction as pc
import numpy as np
from time import time



def calculate_weightsmap_var(noisemap,filter_size=4, nex=1, mask_noisemap = None, fit_gaussian_surface=False, verbose=True):
    '''
    nex =  1: the noisemap was acquired with NEX = 1
    nex = -1: the noisemap was acquired with NEX = 1, but we want to compute the equivalent for a NEX = 2 (we divide by np.sqrt(2))
    nex =  2: the noisemap was acquired with NEX = 2, but we want to compute the equivalent for a NEX = 1 (we multiply by np.sqrt(2))
    '''

    if mask_noisemap is None:
        mask_noisemap = np.ones_like(noisemap.real)

    if nex == 2:
        print 'multiplying noise map by sqrt(2)'
        noisemap = noisemap * np.sqrt(2.)
    elif nex == -1:
        print 'dividing noise map by sqrt(2)'
        noisemap = noisemap / np.sqrt(2.)

    nR, nC, nS = noisemap.shape

    weights_map = np.zeros((nR,nC,nS))

    if verbose==True:
        print 'Weightmap calculation'

    x = range(nR)
    y = range(nC)

    x, y = np.meshgrid(x, y)

    variance_sample_mean_vec = np.zeros((nS,))
    variance_image_mean_vec = np.zeros((nS,))

    stdev_sample_mean_vec = np.zeros((nS,))
    stdev_image_mean_vec = np.zeros((nS,))

    weights_map = np.ones_like(noisemap.real)

    variance_map_hat_global = compute_fast_augmented_var_complex(noisemap,mask_noisemap,filter_size)

    samplestds = np.zeros(nS)
    samplestds_sample = np.zeros(nS)

    for s in xrange(nS):

        variance_map_hat_avg = variance_map_hat_global[:,:,s]
        variance_image_mean_vec = np.mean(variance_map_hat_avg[mask_noisemap[...,s]==1])
        variance_sample_mean_vec = 0.5*(np.var(noisemap.real[mask_noisemap[...,s]==1,s]) + np.var(noisemap.imag[mask_noisemap[...,s]==1,s]))

        samplestds[s] = np.sqrt(variance_image_mean_vec)
        samplestds_sample[s] = np.sqrt(variance_sample_mean_vec)

        #weights_map_sample[:,:,nS]  = (variance_sample_mean_vec[nS] / variance_map_hat_avg)**0.5
        weights_map[:,:,s]   = (variance_image_mean_vec / variance_map_hat_avg)**0.5

        for r in xrange(nR):
            for c in xrange(nC):
                if mask_noisemap[r,c,s]==0:
                    weights_map[r,c,s] = 1.


    if verbose==True:
        print 'done'

    return weights_map, variance_map_hat_global**0.5, samplestds, samplestds_sample

'''
STANDARD DEVIATION FILTER for NOISEMAP
'''

def spherical_footprint(radius):
    fp = np.zeros((radius*2+1,radius*2+1,radius*2+1))
    for x in xrange(radius*2+1):
        for y in xrange(radius*2+1):
            for z in xrange(radius*2+1):
                if (x-radius)**2 + (y-radius)**2+ (z-radius)**2<=radius**2:
                    fp[x,y,z]=1
    return fp

def spherical_footprint_idx(radius):
    '''calculates the indexs of the footprint
       wrt the center
    '''
    sfp = spherical_footprint(radius)
    wres = np.where(sfp ==1)
    n_idxs = len(wres[0])
    sfp_idxs = np.zeros((3,n_idxs),dtype=int)
    for k in xrange(len(wres[0])):
        sfp_idxs[0,k] = wres[0][k]-int(radius)
        sfp_idxs[1,k] = wres[1][k]-int(radius)
        sfp_idxs[2,k] = wres[2][k]-int(radius)

    return sfp_idxs


def check_attempted_location_block(vec,nR):

    upper = vec >=nR
    lower = vec <0

    # mirror criterion
    #vec[upper] = vec[upper]-nR
    #vec[lower] = nR+vec[lower]

    # reflection criterion
    vec[upper] = nR- (vec[upper]-nR)-1
    vec[lower] = -vec[lower]

    mask = upper | lower
    return ~mask,vec


def compute_locations(where_results,footprint_idxs,nR,nC,nS):
    number_of_volumes = footprint_idxs.shape[-1]

    locations = []

    for n in xrange(number_of_volumes):
        wres0 = where_results[0]+footprint_idxs[0,n]
        wres1 = where_results[1]+footprint_idxs[1,n]
        wres2 = where_results[2]+footprint_idxs[2,n]
        mask0,wres0 = check_attempted_location_block(wres0,nR)
        mask1,wres1 = check_attempted_location_block(wres1,nC)
        mask2,wres2 = check_attempted_location_block(wres2,nS)

        mask_location = mask0 & mask1 & mask2

        #locations.append((wres0[mask_location],wres1[mask_location],wres2[mask_location]))
        locations.append((wres0,wres1,wres2))
    return locations,mask_location


def compute_fast_augmented_var_complex(data,mask,radius):

    nR, nC, nS = mask.shape
    std_to_return = np.zeros((nR,nC,nS))

    footprint_idxs = spherical_footprint_idx(radius)

    mask_locations = np.zeros(footprint_idxs.shape[-1],int)
    values_to_use = np.zeros(footprint_idxs.shape[-1]*2)
    locations = footprint_idxs.copy()

    N = np.sum(mask)
    values_to_use = np.zeros((N,footprint_idxs.shape[-1]*2))


    wres = np.where(mask==1)

    locations,mask_location = compute_locations(wres,footprint_idxs,nR, nC, nS)

    mean_volume = np.zeros((nR,nC,nS))
    var_volume = np.zeros((nR,nC,nS))

    for k in xrange(len(locations)):
        mean_volume[wres] = mean_volume[wres] + data.real[locations[k]] + data.imag[locations[k]]


    for k in xrange(len(locations)):
        var_volume[wres] = var_volume[wres] + (mean_volume[wres]/(2*len(locations)) - data.real[locations[k]])**2 + (mean_volume[wres]/(2*len(locations)) - data.imag[locations[k]])**2

    return var_volume/(2*len(locations))


def apc_worker(data_noisy_sv,mask_slice,samplestds_nex_b0s,samplestds_nex_dwis,weightsmap,is_b0,regularization,criterion,pars):

    if is_b0 == True:
        sd_to_give = samplestds_nex_b0s
    else:
        sd_to_give = samplestds_nex_dwis

    res = cmpxadptvreg.adaptive_regularizer(data_noisy_sv,
                                            sd_to_give,
                                            mask_slice,
                                            mask_slice,
                                            weightsmap,
                                            regularization,
                                            criterion,
                                            pars,
                                            minimal_output=False)
    data_smooth = res[0]
    std_dc      = res[2]
    lambdas     = res[5]
    SUREs_hat   = res[7]
    lambda_sure_hat = lambdas[np.argmin(SUREs_hat)]
    return pc.calculate_phase_corrected_image(data_noisy_sv, data_smooth), std_dc,lambda_sure_hat


def parallel_adaptive_phase_correction(data,mask,samplestds_b0s,samplestds_dwis,weightsmap,b0mask,regularization,criterion,search_type,pars,n_workers=1,verbose=True):
    '''
    This version handles
    References:
    [1] Pizzolato et al., (2019). Spatially Varying Monte Carlo SURE for the Regularization of Biomedical Images. ISBI 2019.
    [2] Pizzolato et al., (2019). Adaptive Phase Correction for Diffusion-Weighted Images. NeuroImage.
    '''

    theshape = data.shape
    if len(theshape)==3:
        nR, nC, nB = theshape
        stds = np.zeros(nB)
        lambdas = np.zeros(nB)
        nS = 1
    else:
        nR, nC, nS, nB = theshape
        stds = np.zeros((nS,nB))
        lambdas = np.zeros((nS,nB))

    data_smooth = np.zeros_like(data)

    if n_workers == 0:
        n_workers = mp.cpu_count()
    if verbose==True:
        print 'Parallel adaptive regularization using ' + str(n_workers) + ' workers'

    prev_time = 0.
    for s in xrange(nS):
        if s>0:
            if int((nS-s)*prev_time) < 60:
                toappend = str(int((nS-s)*prev_time )) + ' seconds'
            else:
                toappend = str(int((nS-s)*prev_time /60. )) + ' minutes'
            if verbose==True:
                print 'processing slice ' + str(s) + ' of ' + str(nS) + ' expected time ' + toappend
        else:
            if verbose==True:
                print 'processing slice ' + str(s) + ' of ' + str(nS)

        start = time()

        if nS>1:
            wm2give = weightsmap[...,s]
            sd2give_b0s = samplestds_b0s[s]
            sd2give_dwis = samplestds_dwis[s]
            mask_slice = mask[...,s]
        else:
            wm2give = weightsmap
            sd2give_b0s = samplestds_b0s
            sd2give_dwis = samplestds_dwis
            mask_slice = mask

        pool = mp.Pool(processes=n_workers)
        if nS>1:
            results = [pool.apply_async(apc_worker, args=(data[:,:,s,x],mask_slice, sd2give_b0s,sd2give_dwis,wm2give,b0mask[x],regularization,criterion,pars)) for x in range(nB)]
        else:
            results = [pool.apply_async(apc_worker, args=(data[:,:,x],mask_slice, sd2give_b0s,sd2give_dwis,wm2give,b0mask[x],regularization,criterion,pars)) for x in range(nB)]

        pool.close()
        pool.join()

        if nS>1:
            for b in xrange(nB):
                data_smooth[:,:,s,b] = results[b].get()[0]
                stds[s,b] = results[b].get()[1]
                lambdas[s,b] = results[b].get()[2]
        else:
            for b in xrange(nB):
                data_smooth[:,:,b] = results[b].get()[0]
                stds[b] = results[b].get()[1]
                lambdas[b] = results[b].get()[2]

        prev_time = time() - start

    if verbose==True:
        print 'done'

    return data_smooth, stds, lambdas
