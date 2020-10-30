import numpy as np
from phase_correction import complex_anisotropic_filters as cmpxfilters


def get_defaults_parameters(criterion,search_type=''):
    '''
    parameters = get_defaults_parameters(criterion,search_type):
    Initialize a dictionary called 'parameters' containing the default settings
    based on the adopted criterion (None,'discrepancy','sure') and, in case of
    'sure', the search_type ('increments','bisection').

    it returns the following dictionary of 'parameters'

    parameters['eps']                   # epsilon for numerical stability
    parameters['n_iter_max']            # maximum number of iteration within one regularization cycle (number of iteration done to reach convergence for a given lambda)
    parameters['sigma_peronamalik']     # sigma for Perona-Malik regularization
    parameters['lambda_value']          # lambda value: the default value must be changed if criterion is None
    parameters['epsilon_sure']          # stability parameter for SURE method: it should be as small as possible, until numerical instability occurs. Should not exceed 1.
    parameters['increments']            # values specifying the percentage of lambda found with the discrepancy criterion, in which to look for a SURE solution. If search_type=='increments', each value of 'increments' will be tested.
                                        # If search_type=='bisection', then only the first and last values of 'increments' are used as search space
    parameters['bootstrap_repetitions'] # number of noise realizations used to estimate the "black box" divergence of the operator: useful if the image is small
    parameters['search_type']           # 'increments': find the minimum over a grid lambda_dc*increments. 'bisection': golden bisection rule [1] used in range [lambda_dc*increments[0],lambda_dc*increments[-1]]
    parameters['tolerance_type']        # type of tolerance for stopping criterion in 'bisection' search_type: 'lambda','sure','std'
    parameters['tolerance_value']       # percentage value over tolerance_type: ex. 0.1 is 10% of lambda calculated with discrepancy criterion if search_type=='lambda'
    parameters['max_bisection_iters']   # maximum iterations to perform if tolerance_value is never reached
    parameters['divergence_estimation'] # divergence estimation for SURE: 'stationary' is the standard proven procedure. 'spatially varying' is a variation (not proven)
    parameters['sure_estimator']        # estimating the mean squared error 'mse' or the weighted mse 'wmse'

    [1] Braun, W. J., & Murdoch, D. J. (2007). A first course in statistical programming with R. Cambridge University Press. Pages 132-135.
    '''


    parameters = {}

    parameters['eps']                   = [1e-10,'default']         # epsilon for numerical stability
    parameters['n_iter_max']            = [150,'default']           # maximum number of iteration within one regularization cycle (number of iteration done to reach convergence for a given lambda)
    parameters['sigma_peronamalik']     = [0.,'default']            # sigma for Perona-Malik smoothing
    parameters['lambda_value']          = [1e3,'default']           # lambda value (regularization parameter), used only when criterion is None
    parameters['tol_stop']              = [1e-6,'default']          # used to stop the anisotropic filter before it reaches n_iter_max iteration: stop if the norm of the residuals over 2 consectutive iterations norm_(it)/norm_(it-1) < tol_stop
    parameters['epsilon_sure']          = [[],'not assigned']       # stability parameter for SURE method: it should be as small as possible, until numerical instability occurs. Should not exceed 1.
    parameters['increments']            = [[],'not assigned']       # values specifying the percentage of lambda found with the discrepancy criterion, in which to look for a SURE solution. If search_type=='bisection', then only the first and last values of increments are used as search space
    parameters['bootstrap_repetitions'] = [[],'not assigned']       # number of noise realizations used to estimate the "black box" divergence of the operator: useful if the image is small
    parameters['search_type']           = [[],'not assigned']       # 'increments': find the minimum over a grid lambda_dc*increments. 'bisection': golden bisection rule used in range [lambda_dc*increments[0],lambda_dc*increments[-1]]
    parameters['tolerance_type']        = [[],'not assigned']       # type of tolerance for stopping criterion in 'bisection' search_type: 'lambda','sure','std'
    parameters['tolerance_value']       = [[],'not assigned']       # percentage value over tolerance_type: ex. 0.1 is 10% of lambda calculated with discrepancy criterion if search_type=='lambda'
    parameters['max_bisection_iters']   = [[],'not assigned']       # maximum iterations to perform if tolerance_value is never reached
    parameters['divergence_estimation'] = [[],'not assigned']       # divergence estimation for SURE: 'stationary' is the standard proven procedure. 'spatially varying' is a variation (not proven)
    parameters['sure_estimator']        = [[],'not assigned']       # estimating the mean squared error (mse) or the weighted mse (wmse)
    parameters['weightmap_type']        = [[],'not assigned']       # 'standard_deviation' or 'variance': this is the type of map adotped and is important when 'divergence_estimation' is set to 'spatially_varying'
    parameters['inpainting']            = [False,'default']         # use inpainting method
    parameters['mask_inpainting']       = [[],'not assigned']

    if criterion == 'discrepancy' or criterion == 'sure':
        parameters['lambda_value'] = ['auto','default']

    if criterion == 'sure':
        parameters['epsilon_sure']          = [0.01,'default']
        parameters['bootstrap_repetitions'] = [1,'default']
        parameters['tolerance_type']        = ['lambda','default']
        parameters['tolerance_value']       = [0.01,'default']
        parameters['max_bisection_iters']   = [50,'default']
        parameters['search_type']           = [search_type,'default']
        parameters['sure_estimator']        = ['mse','default']
        parameters['divergence_estimation'] = ['spatially_varying','default']
        parameters['weightmap_type']        = ['standard_deviation','default']
        if search_type == 'increments':
            parameters['increments'] = [np.linspace(1,10.5,100),'default']
        if search_type == 'bisection':
            parameters['increments'] = [np.array([1.,10.5]),'default']

    return parameters

def set_custom_parameter(parameters,key_string,custom_value):
    '''
    parameters = set_custom_parameter(parameters,key_string,custom_value)

    allows setting a parameter to a custom value:
        > parameters = get_defaults_parameters('sure','increments')
        > parameters = set_custom_parameter(parameters,'increments',np.linspace(1,2.5,100))
    '''
    parameters[key_string][0] = custom_value
    parameters[key_string][1] = 'custom'
    return parameters

def print_parameters(parameters):
    '''
    prints the dictionary of parameters generated with get_defaults_parameters
    '''
    for key_string in parameters.keys():
        if parameters[key_string][1] != 'not assigned':
            if key_string == 'search_type':
                print key_string.upper() + ' is set to ' + parameters[key_string][1].upper() + ' value "' + str(parameters[key_string][0]) + '" in ' + str(parameters['increments'][1]).upper() + ' range ' + str(parameters['increments'][0][0]) + '-'+ str(parameters['increments'][0][-1])
            elif key_string != 'increments':
                print key_string.upper() + ' is set to ' + parameters[key_string][1].upper() + ' value "' + str(parameters[key_string][0]) + '"'


def adaptive_regularizer(img,sample_std,mask_slice,mask_slice_eroded,weightmap,regularization, criterion, parameters,minimal_output=False):
    '''
    res = adaptive_regularizer(img,sample_std,mask_slice,mask_slice_eroded,weightmap,regularization, criterion, parameters)

    Adaptive regularizer: this function works as a wrapper around the anisotropic
    diffusion filters, i.e. Laplacian, TV, Perona-Malik.
    It allows for the selection of:
    regularization:     'lap', 'tv',  'pm-gauss' (note that for this 'sigma_peronamalik' has to be specified)
    weightmap:          if specified, this is used to adaptively scale the attachment to data '_lambda', if weightmap==None then no scaling occurs
    criterion:          'discrepancy' [1,2], Monte-Carlo 'sure' (Stein's unbiased robust estimator) [1], 'None', in which case 'lambda_value' has to be specified.
                        When a weightmap is specified, then a 'weighted' variation of Stein's criterion [4] developed in [5] is adopted.
                        When 'sure' is selected, then a first estimate is obtained with the faster discrepancy criterion, and from this point a search of the optimal value according to SURE is performed, see 'search_type' in the help of get_defaults_parameters(criterion,search_type='').

    Other input arguments:
    img:                the input (MxN) image to be regularized
    sample_std:         sample standard deviation estimated on all of the image pixels
    mask_slice:         the MxN mask of the 2D image. This mask includes all of the pixels for there is data (it should not be a brain mask)
    mask_slice_eroded   the MxN eroded (binary erosion) version of the previous mask

    Output:
    if criterion == 'discrepancy' or None, then

    data_smooth_dc   = res[0]       (MxN) regularized image with discrepancy criterion or with specified lambda (if critertion is None)
    lambda_dc        = res[1]       lambda corresponding to that found with discrepancy criterion (or lambda specified by user if criterion is None)
    std_dc           = res[2]       standard deviation estimated by the discrepancy criterion (or that corresponding to the user speficied lambda if critetrion is None)

    if criterion == 'sure', then
    data_smooth_dc   = res[0]
    lambda_dc        = res[1]
    std_dc           = res[2]
    data_smooth_sure = res[3]       (MxN) regularized data with Monte Calro sure criterion
    data_smooth      = res[4]       (MxNxL) regularized images, where L is the number of tested "lambda" values depending on the 'search_type' specified in the dictionary 'parameters'
    lambdas          = res[5]       vector of size (L) containing the tested 'lambda' values
    stds             = res[6]       vector of size (L) containing the estimated standard deviations corresponding to the tested 'lambda' values
    SUREs_hat        = res[7]       vector of size (L) containing the SURE estimates corresponding to the tested 'lambda' values, averaged over all the 'bootstrap_repetitions' specified in the dictionary 'parameters'
    SUREs            = res[8]       matrix of size (L x 'bootstrap_repetitions') containing the SURE estimates for each tested 'lambda' value and for each bootrap estimation of the divergence of the operator

    [1] Morozov, V. A. (1968). The error principle in the solution of operational equations by the regularization method. Zhurnal Vychislitel'noi Matematiki i Matematicheskoi Fiziki, 8(2), 295-309.
    [2] Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. Physica D: nonlinear phenomena, 60(1-4), 259-268.
    [3] Ramani, S., Blu, T., & Unser, M. (2008). Monte-Carlo SURE: A black-box optimization of regularization parameters for general denoising algorithms. IEEE Transactions on Image Processing, 17(9), 1540-1554.
    [4] Stein, C. M. (1981). Estimation of the mean of a multivariate normal distribution. The annals of Statistics, 1135-1151.
    [5] Pizzolato et al., (2019). Spatially Varying Monte Carlo SURE for the Regularization of Biomedical Images. ISBI 2019.
    '''

    eps                   = parameters['eps'][0]
    n_iter_max            = parameters['n_iter_max'][0]
    sigma_peronamalik     = parameters['sigma_peronamalik'][0]
    lambda_value          = parameters['lambda_value'][0]
    tol_stop              = parameters['tol_stop'][0]
    epsilon_sure          = parameters['epsilon_sure'][0]
    increments            = parameters['increments'][0]
    bootstrap_repetitions = parameters['bootstrap_repetitions'][0]
    search_type           = parameters['search_type'][0]
    tolerance_type        = parameters['tolerance_type'][0]
    tolerance_value       = parameters['tolerance_value'][0]
    max_bisection_iters   = parameters['max_bisection_iters'][0]
    divergence_estimation = parameters['divergence_estimation'][0]
    sure_estimator        = parameters['sure_estimator'][0]
    weightmap_type        = parameters['weightmap_type'][0]
    use_inpainting        = parameters['inpainting'][0]
    mask_inpainting       = parameters['mask_inpainting'][0]

    if weightmap is None:
        weightmap = np.ones_like(img)

    single_call = False

    if criterion is None or criterion == 'discrepancy':
        single_call = True

    # override value, here we should just do a check of parameters
    if criterion == 'discrepancy' or criterion == 'sure':
        lambda_value = 'auto'

    is_complex = False
    if type(img.flatten()[0]) == np.complex128:
        is_complex = True

    # This call will always be performed

    if use_inpainting==True:
        img_reg, lambda_iters = cmpxfilters.fp_2d_anisotropic_diffusion_W_fast_inpainting_avg(img,
                                                                            regularization=regularization,
                                                                            _lambda=lambda_value,
                                                                            n_iter_max=n_iter_max,
                                                                            eps=eps,
                                                                            sigma=sigma_peronamalik,
                                                                            sd=sample_std,
                                                                            mask=mask_slice,
                                                                            W=weightmap,
                                                                            tol_stop=tol_stop,
                                                                            mask_inpainting=mask_inpainting)
    else:
        img_reg, lambda_iters = cmpxfilters.fp_2d_anisotropic_diffusion_W_fast(img,
                                                                        regularization=regularization,
                                                                        _lambda=lambda_value,
                                                                        n_iter_max=n_iter_max,
                                                                        eps=eps,
                                                                        sigma=sigma_peronamalik,
                                                                        sd=sample_std,
                                                                        mask=mask_slice,
                                                                        W=weightmap,
                                                                        tol_stop=tol_stop)


    num_of_pixels_mask = (np.sum(mask_slice_eroded)).astype(np.float64)
    mask_slice_eroded_idxs = mask_slice_eroded==1
    d = img_reg[mask_slice_eroded_idxs] - img[mask_slice_eroded_idxs]

    resnorm_2_avg_real = np.sum(d.real**2)/ num_of_pixels_mask#((M-1)*(N-1))
    sd_mean_hat_real   = np.sqrt(resnorm_2_avg_real)
    if is_complex == True:
        resnorm_2_avg_imag = np.sum(d.imag**2)/ num_of_pixels_mask#((M-1)*(N-1))
        sd_mean_hat_imag   = np.sqrt(resnorm_2_avg_imag)
        sd_mean_hat        = (sd_mean_hat_real + sd_mean_hat_imag) / 2.
    else:
        sd_mean_hat        = sd_mean_hat_real

    index_of_last_computed_lambda = int(np.sum(lambda_iters>0))-1
    lambda_dc  = lambda_iters[index_of_last_computed_lambda]
    std_dc     = sd_mean_hat
    img_reg_dc = img_reg

    if single_call:
        if minimal_output == True:
            return img_reg_dc
        else:
            return img_reg_dc, lambda_dc, std_dc, lambda_iters

    # SURE with increments
    max_sure_iterations = len(increments)
    if search_type == 'increments':
        lambdas = lambda_dc*increments

    # SURE with BISECTION
    #   tolerance_value
    #   max_bisection_iters
    if search_type == 'bisection':
        max_sure_iterations = max_bisection_iters # at least 3
        lambdas    = np.zeros(max_bisection_iters)

    stds    = np.zeros_like(lambdas)
    stds_bs = np.zeros((len(lambdas),bootstrap_repetitions) )
    SUREs   = np.zeros((len(lambdas),bootstrap_repetitions) )

    nR, nC =  img_reg_dc.shape
    imgs_reg = np.zeros((nR,nC,len(lambdas)),dtype=np.complex128)


    # Noise to estimate operator divergence 'black box'
    noise_real =  1.* np.random.standard_normal((nR,nC,bootstrap_repetitions))
    noise_imag =  1.* np.random.standard_normal((nR,nC,bootstrap_repetitions))

    if divergence_estimation == 'spatially_varying':
        noise_energy  = 1.*nR*nC#num_of_pixels_mask
        '''
        ABOUT THE MASK MANAGEMENT
        Note: here it is assumed the the whole image is sent to the regularizer,
        therefore, the scaling "noise_energy" is set to the number of pixels in
        the image. This is because, as part of "mask management", the regularize
        r (e.g. TV) function operates on the whole image, but only measures (acc
        ounts for) the MSE (to compute \lambda for the discrepancy criterion, DC)
        within the mask: outside the mask the weightmap is set to 1 (which is
        "wrong"), though within the mask the relative proportions of the standar
        d deviation (e.g. in the center it is higher than on the external corona
        ) are preserved. The function fp_2d_anisotropic_diffusion_W_fast, though
        "imporperly" regularizing according to weighmap=1 outside the mask, only
        uses the results within the mask to calculate the MSE (necessary for the
        estimation of \lambda with DC). Thus, effectively, the samples of noise
        of "noise_real" and "noise_imag" that are located outside the mask (and
        which will be used to calculate SURE and particularly the "divergence"
        later on) are also never used. Nevertheless, when a mask is not specified
        then the mask is set to np.ones((nR,nC)) and all the pixels will have a
        contribution.
        '''

        if weightmap_type == 'variance':
            noise_var = sample_std**2 / weightmap
        if weightmap_type == 'standard_deviation':
            noise_var = sample_std**2 / weightmap**2


        noise_var = noise_var / np.sum(noise_var) * noise_energy
        noise_std = np.sqrt(noise_var)
        for bsrep in xrange(bootstrap_repetitions):
            noise_real[...,bsrep] = noise_std *noise_real[...,bsrep]
            noise_imag[...,bsrep] = noise_std *noise_imag[...,bsrep]


    # for 'increments' and 'bisection'
    cnt = 0
    # for 'bisection'


    if search_type == 'bisection':
        tol = 1.             # 100%
        lambda_min = lambda_dc*increments[0]
        lambda_max = lambda_dc*increments[-1]
        SURE_min   = np.inf
        SURE_max   = np.inf
        computed_sure = ''
        golden_ratio = 2./(np.sqrt(5.) + 1.)

    last_iteration = False

    while (search_type == 'increments' and cnt < len(increments)) or (search_type == 'bisection' and (tol>tolerance_value and cnt < max_sure_iterations)) or last_iteration:

        if search_type == 'increments':
            lambda_value = lambdas[cnt]

        if search_type == 'bisection':
            if cnt < 2:
                # evaluate the first 2 initial points
                lambda_value = lambda_dc*increments[cnt]
                if cnt == 0:
                    lambda_1 = lambda_max - golden_ratio*(lambda_max - lambda_min)
                    lambda_value = lambda_1
                if cnt == 1:
                    lambda_2 = lambda_min + golden_ratio*(lambda_max - lambda_min)
                    lambda_value = lambda_2


        res = regularize_with_lambda_and_compute_sure(epsilon_sure,
                                                      sample_std,
                                                      lambda_value,
                                                      bootstrap_repetitions,
                                                      noise_real,
                                                      noise_imag,
                                                      img,
                                                      mask_slice_eroded_idxs,
                                                      num_of_pixels_mask,
                                                      regularization,
                                                      n_iter_max,
                                                      eps,
                                                      sigma_peronamalik,
                                                      mask_slice,
                                                      weightmap,
                                                      sure_estimator,
                                                      tol_stop,
                                                      use_inpainting,
                                                      mask_inpainting)

        imgs_reg[...,cnt] = res[0]
        SUREs[cnt,:]      = res[1]
        stds[cnt]         = res[2]

        if search_type == 'bisection':
            lambdas[cnt] = lambda_value

            if cnt == 0:
                SURE_1 = np.mean(np.squeeze(SUREs[cnt,:]))
            if cnt == 1:
                SURE_2 = np.mean(np.squeeze(SUREs[cnt,:]))

            if cnt >= 2:
                if computed_sure == 'SURE_1':
                    SURE_1 = np.mean(np.squeeze(SUREs[cnt,:]))
                if computed_sure == 'SURE_2':
                    SURE_2 = np.mean(np.squeeze(SUREs[cnt,:]))

                if (SURE_2 > SURE_1):
                    lambda_max = lambda_2
                    lambda_2 = lambda_1
                    SURE_2 = SURE_1
                    # set new lower test point
                    lambda_1 = lambda_max - golden_ratio*(lambda_max - lambda_min)
                    lambda_value = lambda_1
                    computed_sure = 'SURE_1'
                else:
                    lambda_min = lambda_1
                    lambda_1 = lambda_2
                    SURE_1 = SURE_2
                    # set new upper test point
                    lambda_2 = lambda_min + golden_ratio*(lambda_max - lambda_min)
                    lambda_value = lambda_2
                    computed_sure = 'SURE_2'

            tol_lambda = np.abs(lambda_max - lambda_min) / lambda_dc
            tol_sure   = np.abs(np.mean(np.squeeze(SUREs[cnt,:])) - np.mean(np.squeeze(SUREs[cnt-1,:]))) / np.mean(np.squeeze(SUREs[0,:]))
            tol_stds   = np.abs(stds[cnt]-stds[cnt-1])  / std_dc

            if tolerance_type == 'lambda':
                tol = tol_lambda
            if tolerance_type == 'sure':
                tol = tol_sure
            if tolerance_type == 'std':
                tol = tol_stds

            if last_iteration:
                # we need to reshape the results vectors
                imgs_reg = imgs_reg[...,0:cnt]
                SUREs    = SUREs[0:cnt,:]
                stds     = stds[0:cnt]
                lambdas  = lambdas[0:cnt]
                last_iteration = False
            elif (tol<=tolerance_value) or (cnt >= max_sure_iterations-1):
                last_iteration = True
                lambda_value = (lambda_max + lambda_min) / 2.

        cnt = cnt+1



    SUREs_hat    = np.mean(SUREs,axis=1)
    if search_type == 'increments':
        sure_min_idx = np.argmin(SUREs_hat)
    if search_type == 'bisection':
        sure_min_idx = np.argmin(SUREs_hat)

    img_reg_sure = imgs_reg[:,:,sure_min_idx]

    if minimal_output == True:
        return img_reg_sure
    else:
        return img_reg_dc, lambda_dc, std_dc, img_reg_sure, imgs_reg, lambdas, stds, SUREs_hat, SUREs


def regularize_with_lambda_and_compute_sure(epsilon_sure,sample_std,lambda_value,bootstrap_repetitions,noise_real,noise_imag,img,mask_slice_eroded_idxs,num_of_pixels_mask,regularization,n_iter_max,eps,sigma_peronamalik,mask_slice,weightmap,sure_estimator,tol_stop,use_inpainting,mask_inpainting):

    if use_inpainting==True:
        img_reg, lambda_iters = cmpxfilters.fp_2d_anisotropic_diffusion_W_fast_inpainting_avg(img,
                                                                        regularization=regularization,
                                                                        _lambda=lambda_value,
                                                                        n_iter_max=n_iter_max,
                                                                        eps=eps,
                                                                        sigma=sigma_peronamalik,
                                                                        sd=0.,
                                                                        mask=mask_slice,
                                                                        W=weightmap,
                                                                        tol_stop=tol_stop,
                                                                        mask_inpainting=mask_inpainting)
    else:
        img_reg, lambda_iters = cmpxfilters.fp_2d_anisotropic_diffusion_W_fast(img,
                                                                                regularization=regularization,
                                                                                _lambda=lambda_value,
                                                                                n_iter_max=n_iter_max,
                                                                                eps=eps,
                                                                                sigma=sigma_peronamalik,
                                                                                sd=0.,
                                                                                mask=mask_slice,
                                                                                W=weightmap,
                                                                                tol_stop=tol_stop)

    is_complex = False
    if type(img.flatten()[0]) == np.complex128:
        is_complex = True

    d = img_reg[mask_slice_eroded_idxs] - img[mask_slice_eroded_idxs]

    resnorm_2_avg_real = np.sum(d.real**2)/ num_of_pixels_mask#((M-1)*(N-1))
    sd_mean_hat_real   = np.sqrt(resnorm_2_avg_real)

    if is_complex==True:
        resnorm_2_avg_imag = np.sum(d.imag**2)/ num_of_pixels_mask#((M-1)*(N-1))
        sd_mean_hat_imag   = np.sqrt(resnorm_2_avg_imag)
        sd_mean_hat        = (sd_mean_hat_real + sd_mean_hat_imag) / 2.
        amse_lambda = 0.5*(resnorm_2_avg_real+resnorm_2_avg_imag)
    else:
        sd_mean_hat = sd_mean_hat_real
        amse_lambda = resnorm_2_avg_real

    w_flatten = (weightmap[mask_slice_eroded_idxs]).flatten()

    d = d*w_flatten
    resnorm_2_avg_real_w = np.sum(   (d.real.flatten())**2)/ num_of_pixels_mask
    if is_complex==True:
        resnorm_2_avg_imag_w = np.sum(   (d.imag.flatten())**2)/ num_of_pixels_mask
        amsew_lambda = 0.5*(resnorm_2_avg_real_w+resnorm_2_avg_imag_w)
    else:
        amsew_lambda = resnorm_2_avg_real_w

    # SUREs = np.zeros(bootstrap_repetitions,dtype=np.complex128)
    SUREs = np.zeros(bootstrap_repetitions)
    f_lambda_y = img_reg

    for bsrep in xrange(bootstrap_repetitions):

        # Try the same lambda on the bootsrapped data
        if is_complex==True:
            z = img + epsilon_sure * (noise_real[...,bsrep]+1j*noise_imag[...,bsrep])
        else:
            z = img + epsilon_sure * (noise_real[...,bsrep])

        if use_inpainting==True:
            f_lambda_z, lambda_iters = cmpxfilters.fp_2d_anisotropic_diffusion_W_fast_inpainting_avg(z,
                                                                                regularization=regularization,
                                                                                _lambda=lambda_value,
                                                                                n_iter_max=n_iter_max,
                                                                                eps=eps,
                                                                                sigma=sigma_peronamalik,
                                                                                sd=sample_std,
                                                                                mask=mask_slice,
                                                                                W=weightmap,
                                                                                tol_stop=tol_stop,
                                                                                mask_inpainting=mask_inpainting)
        else:
            f_lambda_z, lambda_iters = cmpxfilters.fp_2d_anisotropic_diffusion_W_fast(z,
                                                                                regularization=regularization,
                                                                                _lambda=lambda_value,
                                                                                n_iter_max=n_iter_max,
                                                                                eps=eps,
                                                                                sigma=sigma_peronamalik,
                                                                                sd=sample_std,
                                                                                mask=mask_slice,
                                                                                W=weightmap,
                                                                                tol_stop=tol_stop)

        diff_f_lambda = f_lambda_z[mask_slice_eroded_idxs] - f_lambda_y[mask_slice_eroded_idxs]

        div_real = (1./(num_of_pixels_mask*epsilon_sure)) * np.dot(noise_real[mask_slice_eroded_idxs,bsrep].flatten(),diff_f_lambda.real.flatten())
        if is_complex==True:
            div_imag = (1./(num_of_pixels_mask*epsilon_sure)) * np.dot(noise_imag[mask_slice_eroded_idxs,bsrep].flatten(),diff_f_lambda.imag.flatten())
            #noise_cmpx  = noise_real[mask_slice_eroded_idxs,bsrep].flatten() + 1j*noise_imag[mask_slice_eroded_idxs,bsrep].flatten()
            #div = (1./(num_of_pixels_mask*epsilon_sure)) * np.vdot(noise_cmpx,diff_f_lambda.flatten())
            div      = 0.5*(div_real+div_imag)
        else:
            div      = div_real

        if sure_estimator == 'mse':
            SUREs[bsrep]      = amse_lambda - sample_std**2 + 2*sample_std**2 * div

        if sure_estimator == 'wmse':
            SUREs[bsrep]      = amsew_lambda - sample_std**2 + 2*sample_std**2 * div


    return img_reg, SUREs, sd_mean_hat
