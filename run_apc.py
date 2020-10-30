import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL" # https://software.intel.com/en-us/node/528380
# or
# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import nibabel as nib
from dipy.core.gradients import gradient_table
import matplotlib.pyplot as plt
import numpy as np
from phase_correction import volume_phase_correction as vpc
import skimage
from skimage import morphology



NEX = 1
num_workers = 8
use_same_bins = True
b0idxforsnr = 45
radius = 4

################################################################################
################################################################################
msg = 'LOADING DATA'
print msg
#-------------------------------------------------------------------------------

pname = './data'
pnameres = './results'

# SCHEME
from dipy.core.gradients import gradient_table
bvecs = np.loadtxt(os.path.join(pname, 'DWI_1_Series0501_WIP_DWI_1_20181101132622_501.bvec'))
bvals = np.loadtxt(os.path.join(pname, 'DWI_1_Series0501_WIP_DWI_1_20181101132622_501.bval'))
gtab = gradient_table(bvals,bvecs=bvecs,big_delta=0.0476,small_delta=0.0276,b0_threshold=0.99)

# LOADING
imgM = nib.load(os.path.join(pname,'DWI_1_Series0501_WIP_DWI_1_20181101132622_501.nii'))
data_M = (imgM.get_data())[...,:-1]
data_Nm = (imgM.get_data())[...,-1]
img = nib.load(os.path.join(pname,'DWI_1_Series0501_WIP_DWI_1_20181101132622_501_ph.nii'))
data_P = (img.get_data())[...,:-1]
data_Np = (img.get_data())[...,-1]


nR,nC,nS,nB = data_M.shape

# CONVERSION to CARTESIAN REPRESENTATION
data = data_M*np.cos(data_P) + 1j * data_M*np.sin(data_P)
noise_map = data_Nm*np.cos(data_Np) + 1j * data_Nm*np.sin(data_Np)


noise_map = noise_map[:,:,20:30]
data = data[:,:,20:30,:]
nR,nC,nS,nB = data.shape
################################################################################
################################################################################
msg = 'CALCULATE A MASK for the NOISE MAP (due to background suppression)'
print msg
#-------------------------------------------------------------------------------


import dipy.segment.mask as dpmask
from scipy.ndimage.filters import generic_filter
thevar  = generic_filter(noise_map.real, np.var, size=2)
masked, mask_whole = dpmask.median_otsu(thevar**0.2, 2, 1, False, dilate=2)

mask = mask_whole.copy()
for s in xrange(nS):
    for k in xrange(radius+1):
        mask[:,:,s] = skimage.morphology.erosion(mask[:,:,s])


plt.figure()
plt.title('A slice of the noise map, with its mask')
plt.subplot(131)
plt.imshow(mask[:,:,5]*np.abs(noise_map[:,:,5]))
plt.subplot(132)
plt.imshow(np.abs(noise_map[:,:,5]))
plt.subplot(133)
plt.imshow(mask[:,:,5])
# plt.show()
plt.savefig(os.path.join(pnameres,'images_noisemapmask.png'))

################################################################################
################################################################################
msg = 'CALCULATE THE WEIGHT MAP for PHASE CORRECTION'
print msg
#-------------------------------------------------------------------------------

weightsmap,std_map_hat, samplestds_filter,samplestds_sample = vpc.calculate_weightsmap_var(noise_map,
                                                                              filter_size=radius,
                                                                              nex=NEX,
                                                                              mask_noisemap = mask,
                                                                              fit_gaussian_surface=False,
                                                                              verbose=True)

#weightsmap,samplestds_filter,samplestds_sample = vpc.calculate_weightsmap(noise_map,
#                                                                          filter_size=5,
#                                                                          nex=NEX,
#                                                                          mask_noisemap = mask,
#                                                                          n_workers=2,
#                                                                          verbose=True)

outImg=nib.Nifti1Image(weightsmap, imgM.affine)
nib.save(outImg, os.path.join(pnameres,'weightsmap_std.nii.gz'))


# Compute SNR
plt.figure()
plt.imshow(np.abs(data[:,:,5,b0idxforsnr])/np.abs(samplestds_filter[5]/weightsmap[:,:,5]),clim=(0,50))
plt.colorbar()
plt.title('SNR map (slice 25)')
# plt.show()
plt.savefig(os.path.join(pnameres,'images_snrmap.png'))

# show the calculated average standard deviation per slice
samplestds_dwis = samplestds_filter
samplestds_b0s = samplestds_filter / np.sqrt(2)

plt.figure()
plt.plot(samplestds_dwis,label='sd for DWIs')
plt.plot(samplestds_b0s,label='sd for b0s')
plt.xlabel('Axial slice number')
plt.title('SD (signal units)')
plt.title('SD per slice (within mask)')
plt.legend()
# plt.show()
plt.savefig(os.path.join(pnameres,'images_stds_per_slice.png'))

# show the weightsmap
plt.figure()
plt.imshow(weightsmap[:,:,5])
plt.colorbar()
plt.title('Weightmap (slice 25)')
# plt.show()
plt.savefig(os.path.join(pnameres,'images_weightmap.png'))

################################################################################
################################################################################
msg = 'PHASE CORRECTION'
print msg
#-------------------------------------------------------------------------------

# Prepare a b0 mask used for deciding which sd to use (b0s [/sqrt(2)] or DWIs)
b0mask = gtab.b0s_mask

# Select the parameters
from phase_correction import complex_adaptive_regularization as cmpxadptvreg
from time import time

regularization = 'tv'

criterion = 'sure'
#criterion = 'discrepancy'
search_type = 'bisection'

# Get default parameters
pars = cmpxadptvreg.get_defaults_parameters(criterion,
                                            search_type)
# SET custom parameters
pars = cmpxadptvreg.set_custom_parameter(pars,'increments',np.array([0.5,10.5]))
pars = cmpxadptvreg.set_custom_parameter(pars,'bootstrap_repetitions',1)
pars = cmpxadptvreg.set_custom_parameter(pars,'n_iter_max',150)
pars = cmpxadptvreg.set_custom_parameter(pars,'tolerance_value',0.01)#0.01
pars = cmpxadptvreg.set_custom_parameter(pars,'tol_stop',0.000001)



start = time()
data_pc, stds, lambdas_hat = vpc.parallel_adaptive_phase_correction(data[:,:,:,:],mask[:,:,:],
                                                                      samplestds_b0s[:],samplestds_dwis[:],
                                                                      (weightsmap[:,:,:]),
                                                                      b0mask,
                                                                      regularization,criterion,search_type,pars,
                                                                      n_workers=num_workers,verbose=True)
print time()-start

# Save the data
outImg=nib.Nifti1Image(data_pc.real, imgM.affine)
nib.save(outImg, os.path.join(pnameres,'data_pc_real.nii.gz'))
outImg=nib.Nifti1Image(data_pc.imag, imgM.affine)
nib.save(outImg, os.path.join(pnameres,'data_pc_imag.nii.gz'))

np.save(os.path.join(pnameres,'stds.npy'),stds)
np.save(os.path.join(pnameres,'lambdas_hat.npy'),lambdas_hat)

# Check the Sd a posteriori
plt.figure()
plt.plot(samplestds_dwis,label='a priori')
plt.plot(np.mean(stds,axis=1),'--',label='a posteriori')
plt.legend()
plt.title('Check for discrepancy criterion')
# plt.show()
plt.savefig(os.path.join(pnameres,'images_sd_check_discrepancy.png'))







################################################################################
################################################################################
msg = 'COMPUTE HISTOGRAMS'
print msg
#-------------------------------------------------------------------------------

from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter

def minor_tick(x, pos):
    if not x % 1.0:
        return ""
    return "%.2f" % x

def compute_aspect(ax):
    y = ax.get_ylim()
    x = ax.get_xlim()
    return (x[1]-x[0]) / (y[1]-y[0])


## colors
c0 = np.array([147.,208.,191.])/255.
c1 = np.array([64.,149.,181.])/255.
c2 = np.array([55.,83.,94.])/255.

# Region of interest
roi = np.zeros((nR,nC,nS))
s_half = int(nS/2)
roi[30:nR-30,30:nC-30,s_half-2:s_half+2] = 1

for bvalue in np.unique(bvals):
    if bvalue >0:
        selected_imagdwis  = data_pc[:,:,:,bvals==bvalue].imag

        ## figure settings
        figsize = 8.
        fig = plt.figure(figsize=(figsize, figsize*0.7))

        ax = fig.add_subplot(1, 1, 1, aspect=1)
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_major_locator(MultipleLocator(500))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick))
        # ax.set_xlim(0, 4)
        # ax.set_ylim(0, 2000)

        ax.tick_params(which='major', width=1.0)
        ax.tick_params(which='major', length=10, labelsize=15)
        ax.ticklabel_format(axis='y',rotation=70)
        ax.tick_params(which='minor', width=1.0, labelsize=14)
        ax.tick_params(which='minor', length=5, labelsize=14, labelcolor='0.25')
        # ax.tick_params(which='major',labelsize=15)

        ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)


        ## histogram of phase-corrected imaginary part
        XX = selected_imagdwis[roi>0,:].flatten()
        n, bins, patches = ax.hist(XX,normed=True,bins=20,
                                    facecolor=c1,color=c2,
                                    alpha=0.75,rwidth=0.9,
                                    histtype='stepfilled',
                                    lw=4,label='imag (APC) DC',
                                    zorder=1)


        XX = np.hstack((noise_map.real[roi>0.],noise_map.imag[roi>0.]))
        XX = XX*np.sqrt(NEX)

        bins2give=20
        if use_same_bins:
            bins2give=bins

        n, bins, patches = ax.hist(XX,normed=True,bins=bins2give,
                                    facecolor='black',color='black',
                                    alpha=0.99,rwidth=0.85,
                                    histtype='step',
                                    lw=4,label='Noise map',
                                    zorder=2)


        #ax.set_title("Imag PC vs Noise: b-shell ="+ str(int(bvalue)) , fontsize=20, verticalalignment='bottom')
        ax.set_xlabel("signal intensity", fontsize=20)
        ax.set_ylabel("#", fontsize=20)

        ax.set_aspect(compute_aspect(ax)*0.7)
        ax.legend(fontsize=16,loc='best')


        fig_name = 'images_histogram_comparison_bshell_'+ str(int(bvalue)) + '.png'

        plt.title('Imag PC vs Noise: b-shell =' + str(int(bvalue)) , fontsize=20)
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(pnameres,fig_name),dpi=300)
