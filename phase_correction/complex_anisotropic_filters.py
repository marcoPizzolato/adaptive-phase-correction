# -*- coding: utf-8 -*-
import numpy as np
import scipy.ndimage as ndimg



def differences(A,axis):

    A = np.asanyarray(A)
    nd = len(A.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    D = A[slice1] - A[slice2]

    # Forward
    Df = D[slice1]
    # Backward
    Db = D[slice2]
    # Central
    Dc = (Df+Db)/2.0

    return Df, Db, Dc





def fp_2d_anisotropic_diffusion_W_fast(u0, regularization, _lambda=0.028, n_iter_max=200, eps=0.000001, sigma=0.,sd=0.,mask=[0], W=[1],tol_stop=0.000001):
    '''
    W has to be a matrix that scales l, where W has maximum entry 1 and minimum 0
    '''
    M,N = u0.shape
    h = 1.0
    u = u0.copy()


    is_complex = type(u0.flatten()[0])==np.complex128
    is_mask = True
    if len(mask)==1:
        is_mask=False

    _l = _lambda
    #_l0 = 0.5
    #_l0 = 0.707/(sd*np.sqrt(mask.sum())) + 0.6849/(sd**2 *mask.sum())
    #_l0 = 0.707/sd + 0.6849/sd**2
    _l0 = 0.
    if _lambda == 'auto':
        _l0 = 3*0.707/sd + 3*0.6849/sd**2


    _ls = np.zeros(n_iter_max)
    _sn = np.zeros(n_iter_max)

    squared_norm_old = np.inf
    stop_criterion_reached = False
    it = 0
    while it < n_iter_max and not stop_criterion_reached:

        ux_f, ux_b, ux_c = differences(u,0)
        uy_f, uy_b, uy_c = differences(u,1)

        # check Holomorfism, Cauchy-Riemann equations
        #if u.dtype == 'complex128':
        #    print np.linalg.norm(ux_c[:,1:-1].real - uy_c[1:-1,:].imag), np.linalg.norm(ux_c[:,1:-1].real)


        c1 = 1. / np.sqrt(eps*eps + ux_f[:,1:-1].real**2 + uy_c[1:-1,:].real**2 + ux_f[:,1:-1].imag**2 + uy_c[1:-1,:].imag**2)
        c2 = 1. / np.sqrt(eps*eps + ux_b[:,1:-1].real**2 + uy_c[0:-2,:].real**2 + ux_b[:,1:-1].imag**2 + uy_c[0:-2,:].imag**2)
        c3 = 1. / np.sqrt(eps*eps + ux_c[:,1:-1].real**2 + uy_f[1:-1,:].real**2 + ux_c[:,1:-1].imag**2 + uy_f[1:-1,:].imag**2)
        c4 = 1. / np.sqrt(eps*eps + ux_c[:,0:-2].real**2 + uy_b[1:-1,:].real**2 + ux_c[:,0:-2].imag**2 + uy_b[1:-1,:].imag**2)

        ux_c_re = ux_c[:,1:-1].real
        ux_c_im = ux_c[:,1:-1].imag
        uy_c_re = uy_c[1:-1,:].real
        uy_c_im = uy_c[1:-1,:].imag


        norm_of_gradient = np.sqrt(eps*eps + ux_c_re**2 + uy_c_re**2 + ux_c_im**2 + uy_c_im**2)
        #norm_of_gradient = np.sqrt(eps*eps + ux_c[:,1:-1]**2 + uy_c[1:-1,:]**2)

        #print np.linalg.norm(norm_of_gradient.imag)

        if regularization == 'tv':
            # Total Variation
            A = 1. / norm_of_gradient
            B = 0.
        elif regularization == 'lap':
            # Laplacian
            A = 1.
            B = 1.
        elif regularization == 'mcf':
            # Mean curvature flow
            A = 1.
            B = 0.
        elif regularization == 'pm-gauss':
            # Perona-Malik with Gaussian kernel
            A = 1.0
            B = np.exp(-norm_of_gradient**2 / (2.*sigma**2))
        elif regularization == 'transfer':
            # Perona-Malik with Gaussian kernel
            A = 0.0
            B = norm_of_gradient**2
        else:
            raise ValueError('wrong regularization specified')

        if _lambda == 'auto': #and regularization == 'tv':
            if it==0:
                _l = _l0
            else:


                if is_mask:
                    dr = (u[1:-1,1:-1].real-u0[1:-1,1:-1].real)*mask[1:-1,1:-1] #/ (W[1:-1,1:-1])
                    di = (u[1:-1,1:-1].imag-u0[1:-1,1:-1].imag)*mask[1:-1,1:-1] #/ (W[1:-1,1:-1])
                    number_of_pixels = mask.sum()
                else:
                    dr = u[1:-1,1:-1].real-u0[1:-1,1:-1].real #/ (W[1:-1,1:-1])
                    di = u[1:-1,1:-1].imag-u0[1:-1,1:-1].imag #/ (W[1:-1,1:-1])
                    number_of_pixels = (M-1)*(N-1)



                #rmse = np.sum( np.sqrt(dr**2 + di**2) ) / np.sqrt(number_of_pixels)
                #_l *= rmse/sd

                rmse_r = np.sqrt(np.sum(dr**2)/ number_of_pixels)
                rmse_i = np.sqrt(np.sum(di**2)/ number_of_pixels)

                #rmse_r = np.sqrt(np.sum(dr**2) / np.sum(W[1:-1,1:-1]))
                #rmse_i = np.sqrt(np.sum(di**2) / np.sum(W[1:-1,1:-1]))


                if is_complex:
                    _l *= np.mean(np.array([rmse_r/sd,rmse_i/sd]))
                else:
                    _l *= rmse_r/sd

        _ls[it]=_l

        Num = (
                u0[1:-1,1:-1] + (1/(2*_l*W[1:-1,1:-1]*h**2)) *
                (B*(u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]) +
                (A-B) * norm_of_gradient * (c1 * u[2:, 1:-1] + c2 * u[:-2, 1:-1] + c3 * u[1:-1, 2:] + c4 * u[1:-1, :-2]))
        )

        Den = (
                1. + (1/(2*_l*W[1:-1,1:-1]*h**2)) *
                (B*4. +
                (A-B) * norm_of_gradient * (c1 + c2 + c3 + c4))
        )


        u[1:-1,1:-1] = Num / Den

        #print it
        it = it+1

        # CHECK RATE OF CHANGE:
        dr_roc = (u[1:-1,1:-1].real-u0[1:-1,1:-1].real)*mask[1:-1,1:-1]
        squared_norm = np.sum(dr_roc**2)

        _sn[it-1] = squared_norm/squared_norm_old
        if np.abs(squared_norm/squared_norm_old - 1.) < tol_stop:
            stop_criterion_reached = True
        # if it>5:
        #     if np.abs(_sn[it-4] - 1.) < tol_stop:
        #         stop_criterion_reached = True
        squared_norm_old = squared_norm

        u[:,0] = u[:,1]
        u[:,-1] = u[:,-2]
        u[0,:] = u[1,:]
        u[-1,:] = u[-2,:]

        u[0,0] = u[1,1]
        u[0,-1] = u[1,-2]
        u[-1,0] = u[-2,1]
        u[-1,-1] = u[-2,-2]

    return u, _ls#, _sn



def fp_2d_anisotropic_diffusion_W_fast_inpainting(u0, regularization, _lambda=0.028, n_iter_max=200, eps=0.000001, sigma=0.,sd=0.,mask=[0], W=[1],tol_stop=0.000001, mask_inpainting = [3]):
    '''
    W has to be a matrix that scales l, where W has maximum entry 1 and minimum 0
    '''
    M,N = u0.shape
    h = 1.0
    u = u0.copy()


    is_complex = type(u0.flatten()[0])==np.complex128
    is_mask = True
    if len(mask)==1:
        is_mask=False

    if len(mask_inpainting)==1:
        mask_inpainting = np.ones((M,N))
    else:
        mask = mask*mask_inpainting
        is_mask = True
        W = W*mask_inpainting



    _l = _lambda
    #_l0 = 0.5
    #_l0 = 0.707/(sd*np.sqrt(mask.sum())) + 0.6849/(sd**2 *mask.sum())
    #_l0 = 0.707/sd + 0.6849/sd**2
    _l0 = 0.
    if _lambda == 'auto':
        _l0 = 3*0.707/sd + 3*0.6849/sd**2


    _ls = np.zeros(n_iter_max)
    _sn = np.zeros(n_iter_max)

    squared_norm_old = np.inf
    stop_criterion_reached = False
    it = 0
    while it < n_iter_max and not stop_criterion_reached:

        ux_f, ux_b, ux_c = differences(u,0)
        uy_f, uy_b, uy_c = differences(u,1)

        # check Holomorfism, Cauchy-Riemann equations
        #if u.dtype == 'complex128':
        #    print np.linalg.norm(ux_c[:,1:-1].real - uy_c[1:-1,:].imag), np.linalg.norm(ux_c[:,1:-1].real)


        c1 = 1. / np.sqrt(eps*eps + ux_f[:,1:-1].real**2 + uy_c[1:-1,:].real**2 + ux_f[:,1:-1].imag**2 + uy_c[1:-1,:].imag**2)
        c2 = 1. / np.sqrt(eps*eps + ux_b[:,1:-1].real**2 + uy_c[0:-2,:].real**2 + ux_b[:,1:-1].imag**2 + uy_c[0:-2,:].imag**2)
        c3 = 1. / np.sqrt(eps*eps + ux_c[:,1:-1].real**2 + uy_f[1:-1,:].real**2 + ux_c[:,1:-1].imag**2 + uy_f[1:-1,:].imag**2)
        c4 = 1. / np.sqrt(eps*eps + ux_c[:,0:-2].real**2 + uy_b[1:-1,:].real**2 + ux_c[:,0:-2].imag**2 + uy_b[1:-1,:].imag**2)

        ux_c_re = ux_c[:,1:-1].real
        ux_c_im = ux_c[:,1:-1].imag
        uy_c_re = uy_c[1:-1,:].real
        uy_c_im = uy_c[1:-1,:].imag


        norm_of_gradient = np.sqrt(eps*eps + ux_c_re**2 + uy_c_re**2 + ux_c_im**2 + uy_c_im**2)
        #norm_of_gradient = np.sqrt(eps*eps + ux_c[:,1:-1]**2 + uy_c[1:-1,:]**2)

        #print np.linalg.norm(norm_of_gradient.imag)

        if regularization == 'tv':
            # Total Variation
            A = 1. / norm_of_gradient
            B = 0.
        elif regularization == 'lap':
            # Laplacian
            A = 1.
            B = 1.
        elif regularization == 'mcf':
            # Mean curvature flow
            A = 1.
            B = 0.
        elif regularization == 'pm-gauss':
            # Perona-Malik with Gaussian kernel
            A = 1.0
            B = np.exp(-norm_of_gradient**2 / (2.*sigma**2))
        elif regularization == 'transfer':
            # Perona-Malik with Gaussian kernel
            A = 0.0
            B = norm_of_gradient**2
        else:
            raise ValueError('wrong regularization specified')

        if _lambda == 'auto': #and regularization == 'tv':
            if it==0:
                _l = _l0
            else:


                if is_mask:
                    dr = (u[1:-1,1:-1].real-u0[1:-1,1:-1].real)*mask[1:-1,1:-1] #/ (W[1:-1,1:-1])
                    di = (u[1:-1,1:-1].imag-u0[1:-1,1:-1].imag)*mask[1:-1,1:-1] #/ (W[1:-1,1:-1])
                    number_of_pixels = mask.sum()
                else:
                    dr = u[1:-1,1:-1].real-u0[1:-1,1:-1].real #/ (W[1:-1,1:-1])
                    di = u[1:-1,1:-1].imag-u0[1:-1,1:-1].imag #/ (W[1:-1,1:-1])
                    number_of_pixels = (M-1)*(N-1)



                #rmse = np.sum( np.sqrt(dr**2 + di**2) ) / np.sqrt(number_of_pixels)
                #_l *= rmse/sd

                rmse_r = np.sqrt(np.sum(dr**2)/ number_of_pixels)
                rmse_i = np.sqrt(np.sum(di**2)/ number_of_pixels)

                #rmse_r = np.sqrt(np.sum(dr**2) / np.sum(W[1:-1,1:-1]))
                #rmse_i = np.sqrt(np.sum(di**2) / np.sum(W[1:-1,1:-1]))


                if is_complex:
                    _l *= np.mean(np.array([rmse_r/sd,rmse_i/sd]))
                else:
                    _l *= rmse_r/sd

        _ls[it]=_l

        Num = (
                u0[1:-1,1:-1] + (1/(2*_l*W[1:-1,1:-1]*h**2 + eps)) *
                (B*(u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]) +
                (A-B) * norm_of_gradient * (c1 * u[2:, 1:-1] + c2 * u[:-2, 1:-1] + c3 * u[1:-1, 2:] + c4 * u[1:-1, :-2]))
        )

        Den = (
                1. + (1/(2*_l*W[1:-1,1:-1]*h**2 + eps)) *
                (B*4. +
                (A-B) * norm_of_gradient * (c1 + c2 + c3 + c4))
        )


        u[1:-1,1:-1] = Num / Den

        #print it
        it = it+1

        # CHECK RATE OF CHANGE:
        dr_roc = (u[1:-1,1:-1].real-u0[1:-1,1:-1].real)*mask[1:-1,1:-1]
        squared_norm = np.sum(dr_roc**2)

        _sn[it-1] = squared_norm/squared_norm_old
        if np.abs(squared_norm/squared_norm_old - 1.) < tol_stop:
            stop_criterion_reached = True
        # if it>5:
        #     if np.abs(_sn[it-4] - 1.) < tol_stop:
        #         stop_criterion_reached = True
        squared_norm_old = squared_norm

        u[:,0] = u[:,1]
        u[:,-1] = u[:,-2]
        u[0,:] = u[1,:]
        u[-1,:] = u[-2,:]

        u[0,0] = u[1,1]
        u[0,-1] = u[1,-2]
        u[-1,0] = u[-2,1]
        u[-1,-1] = u[-2,-2]

    return u, _ls#, _sn



def fp_2d_anisotropic_diffusion_W_fast_inpainting_avg(u0, regularization, _lambda=0.028, n_iter_max=200, eps=0.000001, sigma=0.,sd=0.,mask=[0], W=[1],tol_stop=0.000001, mask_inpainting = [3]):
    '''
    W has to be a matrix that scales l, where W has maximum entry 1 and minimum 0
    '''
    M,N = u0.shape
    h = 1.0
    u = u0.copy()


    is_complex = type(u0.flatten()[0])==np.complex128
    is_mask = True
    if len(mask)==1:
        raise ValueError('please provide a mask')

    if len(mask_inpainting)==1:
        raise ValueError('inpainting needs an inpainting domain (mask_inpainting)')
    else:
        is_mask = True
        mask1 = mask*mask_inpainting
        W1 = W*mask_inpainting
        u1 = u*mask_inpainting
        mask_inpainting_reciprocal = np.abs(mask_inpainting-1)
        mask2 = mask*mask_inpainting_reciprocal
        W2 = W*mask_inpainting_reciprocal
        u2 = u*mask_inpainting_reciprocal

    _l = _lambda
    _l0 = 0.
    if _lambda == 'auto':
        _l0 = 3*0.707/sd + 3*0.6849/sd**2


    #_ls1 = np.zeros(n_iter_max)
    #_sn1 = np.zeros(n_iter_max)
    #_ls2 = np.zeros(n_iter_max)
    #_sn2 = np.zeros(n_iter_max)
    _ls = np.zeros(n_iter_max)
    _sn = np.zeros(n_iter_max)

    squared_norm_old = np.inf
    stop_criterion_reached = False
    it = 0

    while it < n_iter_max and not stop_criterion_reached:

        ux_f1, ux_b1, ux_c1 = differences(u1,0)
        uy_f1, uy_b1, uy_c1 = differences(u1,1)
        c11 = 1. / np.sqrt(eps*eps + ux_f1[:,1:-1].real**2 + uy_c1[1:-1,:].real**2 + ux_f1[:,1:-1].imag**2 + uy_c1[1:-1,:].imag**2)
        c21 = 1. / np.sqrt(eps*eps + ux_b1[:,1:-1].real**2 + uy_c1[0:-2,:].real**2 + ux_b1[:,1:-1].imag**2 + uy_c1[0:-2,:].imag**2)
        c31 = 1. / np.sqrt(eps*eps + ux_c1[:,1:-1].real**2 + uy_f1[1:-1,:].real**2 + ux_c1[:,1:-1].imag**2 + uy_f1[1:-1,:].imag**2)
        c41 = 1. / np.sqrt(eps*eps + ux_c1[:,0:-2].real**2 + uy_b1[1:-1,:].real**2 + ux_c1[:,0:-2].imag**2 + uy_b1[1:-1,:].imag**2)

        ux_c_re1 = ux_c1[:,1:-1].real
        ux_c_im1 = ux_c1[:,1:-1].imag
        uy_c_re1 = uy_c1[1:-1,:].real
        uy_c_im1 = uy_c1[1:-1,:].imag

        norm_of_gradient1 = np.sqrt(eps*eps + ux_c_re1**2 + uy_c_re1**2 + ux_c_im1**2 + uy_c_im1**2)


        ux_f2, ux_b2, ux_c2 = differences(u2,0)
        uy_f2, uy_b2, uy_c2 = differences(u2,1)
        c12 = 1. / np.sqrt(eps*eps + ux_f2[:,1:-1].real**2 + uy_c2[1:-1,:].real**2 + ux_f2[:,1:-1].imag**2 + uy_c2[1:-1,:].imag**2)
        c22 = 1. / np.sqrt(eps*eps + ux_b2[:,1:-1].real**2 + uy_c2[0:-2,:].real**2 + ux_b2[:,1:-1].imag**2 + uy_c2[0:-2,:].imag**2)
        c32 = 1. / np.sqrt(eps*eps + ux_c2[:,1:-1].real**2 + uy_f2[1:-1,:].real**2 + ux_c2[:,1:-1].imag**2 + uy_f2[1:-1,:].imag**2)
        c42 = 1. / np.sqrt(eps*eps + ux_c2[:,0:-2].real**2 + uy_b2[1:-1,:].real**2 + ux_c2[:,0:-2].imag**2 + uy_b2[1:-1,:].imag**2)

        ux_c_re2 = ux_c2[:,1:-1].real
        ux_c_im2 = ux_c2[:,1:-1].imag
        uy_c_re2 = uy_c2[1:-1,:].real
        uy_c_im2 = uy_c2[1:-1,:].imag

        norm_of_gradient2 = np.sqrt(eps*eps + ux_c_re2**2 + uy_c_re2**2 + ux_c_im2**2 + uy_c_im2**2)


        if regularization == 'tv':
            # Total Variation
            A1 = 1. / norm_of_gradient1
            B1 = 0.
            A2 = 1. / norm_of_gradient2
            B2 = 0.
        elif regularization == 'lap':
            # Laplacian
            A1 = 1.
            B1 = 1.
            A2 = 1.
            B2 = 1.
        elif regularization == 'mcf':
            # Mean curvature flow
            A1 = 1.
            B1 = 0.
            A2 = 1.
            B2 = 0.
        elif regularization == 'pm-gauss':
            # Perona-Malik with Gaussian kernel
            A1 = 1.0
            B1 = np.exp(-norm_of_gradient1**2 / (2.*sigma**2))
            A2 = 1.0
            B2 = np.exp(-norm_of_gradient2**2 / (2.*sigma**2))
        elif regularization == 'transfer':
            # Perona-Malik with Gaussian kernel
            A1 = 0.0
            B1 = norm_of_gradient1**2
            A2 = 0.0
            B2 = norm_of_gradient2**2
        else:
            raise ValueError('wrong regularization specified')

        if _lambda == 'auto': #and regularization == 'tv':
            if it==0:
                _l = _l0
            else:

                dr1 = (u1[1:-1,1:-1].real-u0[1:-1,1:-1].real)*mask1[1:-1,1:-1] #/ (W[1:-1,1:-1])
                di1 = (u1[1:-1,1:-1].imag-u0[1:-1,1:-1].imag)*mask1[1:-1,1:-1] #/ (W[1:-1,1:-1])
                number_of_pixels1 = mask1.sum()

                dr2 = (u2[1:-1,1:-1].real-u0[1:-1,1:-1].real)*mask2[1:-1,1:-1] #/ (W[1:-1,1:-1])
                di2 = (u2[1:-1,1:-1].imag-u0[1:-1,1:-1].imag)*mask2[1:-1,1:-1] #/ (W[1:-1,1:-1])
                number_of_pixels2 = mask2.sum()

                rmse_r1 = np.sqrt(np.sum(dr1**2)/ number_of_pixels1)
                rmse_i1 = np.sqrt(np.sum(di1**2)/ number_of_pixels1)
                rmse_r2 = np.sqrt(np.sum(dr2**2)/ number_of_pixels2)
                rmse_i2 = np.sqrt(np.sum(di2**2)/ number_of_pixels2)
                #MAKE THE MEAN AS OPTION
                #rmse_r = 0.5*(rmse_r1+rmse_r2)
                #rmse_i = 0.5*(rmse_i1+rmse_i2)

                rmse_r = np.sqrt( (np.sum(dr1**2)+np.sum(dr2**2))/ (number_of_pixels1+number_of_pixels2) )
                rmse_i = np.sqrt( (np.sum(di1**2)+np.sum(di2**2))/ (number_of_pixels1+number_of_pixels2) )

                if is_complex:
                    _l *= np.mean(np.array([rmse_r/sd,rmse_i/sd]))
                else:
                    _l *= rmse_r/sd

        _ls[it]=_l

        Num1 = (
                u0[1:-1,1:-1]*mask1[1:-1,1:-1] + (1/(2*_l*W1[1:-1,1:-1]*h**2 + eps)) *
                (B1*(u1[2:, 1:-1] + u1[:-2, 1:-1] + u1[1:-1, 2:] + u1[1:-1, :-2]) +
                (A1-B1) * norm_of_gradient1 * (c11 * u1[2:, 1:-1] + c21 * u1[:-2, 1:-1] + c31 * u1[1:-1, 2:] + c41 * u1[1:-1, :-2]))
        )

        Den1 = (
                1. + (1/(2*_l*W1[1:-1,1:-1]*h**2 + eps)) *
                (B1*4. +
                (A1-B1) * norm_of_gradient1 * (c11 + c21 + c31 + c41))
        )

        Num2 = (
                u0[1:-1,1:-1]*mask2[1:-1,1:-1] + (1/(2*_l*W2[1:-1,1:-1]*h**2 + eps)) *
                (B2*(u2[2:, 1:-1] + u2[:-2, 1:-1] + u2[1:-1, 2:] + u2[1:-1, :-2]) +
                (A2-B2) * norm_of_gradient2 * (c12 * u2[2:, 1:-1] + c22 * u2[:-2, 1:-1] + c32 * u2[1:-1, 2:] + c42 * u2[1:-1, :-2]))
        )

        Den2 = (
                1. + (1/(2*_l*W2[1:-1,1:-1]*h**2 + eps)) *
                (B2*4. +
                (A2-B2) * norm_of_gradient2 * (c12 + c22 + c32 + c42))
        )

        u1[1:-1,1:-1] = Num1 / Den1
        u2[1:-1,1:-1] = Num2 / Den2

        #print it
        it = it+1

        # CHECK RATE OF CHANGE:
        dr_roc1 = (u1[1:-1,1:-1].real-u0[1:-1,1:-1].real)*mask1[1:-1,1:-1]
        dr_roc2 = (u2[1:-1,1:-1].real-u0[1:-1,1:-1].real)*mask2[1:-1,1:-1]
        squared_norm = np.sum(dr_roc1**2)+np.sum(dr_roc2**2)

        _sn[it-1] = squared_norm/squared_norm_old
        if np.abs(squared_norm/squared_norm_old - 1.) < tol_stop:
            stop_criterion_reached = True
        # if it>5:
        #     if np.abs(_sn[it-4] - 1.) < tol_stop:
        #         stop_criterion_reached = True
        squared_norm_old = squared_norm

        u1[:,0] = u1[:,1]
        u1[:,-1] = u1[:,-2]
        u1[0,:] = u1[1,:]
        u1[-1,:] = u1[-2,:]

        u1[0,0] = u1[1,1]
        u1[0,-1] = u1[1,-2]
        u1[-1,0] = u1[-2,1]
        u1[-1,-1] = u1[-2,-2]

        u2[:,0] = u2[:,1]
        u2[:,-1] = u2[:,-2]
        u2[0,:] = u2[1,:]
        u2[-1,:] = u2[-2,:]

        u2[0,0] = u2[1,1]
        u2[0,-1] = u2[1,-2]
        u2[-1,0] = u2[-2,1]
        u2[-1,-1] = u2[-2,-2]

    u = 0.5*(u1+u2)

    return u, _ls#, _sn
