import numpy as np

def calculate_phase_corrected_image(original_image, regularized_image):
    '''
    phase_corrected_image = calculate_phase_corrected_image(original_image, regularized_image)

    inputs:
    original_image          (MxN) noisy complex image
    regularized_image       (MxN) regularized complex image

    output:
    phase_corrected_image   (MxN) phase-corrected complex image
    '''

    estimated_phase = np.angle(regularized_image)
    original_magnitude = np.absolute(original_image)
    original_phase = np.angle(original_image)

    # rotation
    phase_corrected_image = original_magnitude*np.exp(1j*(original_phase-estimated_phase))

    return phase_corrected_image
