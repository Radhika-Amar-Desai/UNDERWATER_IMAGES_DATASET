import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize_of_image(image):
    
    min_val = np.min(image)
    max_val = np.max(image)
    range_val = max_val - min_val
    normalized_image = (image - min_val) / range_val
    normalized_image = (normalized_image * 255).astype(np.uint8)

    return normalized_image

def log_of_image(image):

    normalized_image = image / 255.0
    log_image = np.log(normalized_image + 1)  # Add 1 to avoid log(0)
    rescaled_image = log_image * 255.0
    rescaled_image = rescaled_image.astype(np.uint8)

    return rescaled_image

def DFT_of_image(image):

    image_fft = np.fft.fft2(image)
    image_fft_shifted = np.fft.fftshift(image_fft)
    magnitude_spectrum = np.abs(image_fft_shifted)
    phase_spectrum = np.angle(image_fft_shifted)
    
    return magnitude_spectrum, phase_spectrum

def update_radius(value):
    global radius
    radius = value
    apply_filters()

def update_gamma_L(value):
    global gamma_L
    gamma_L = value / 100
    apply_filters()

def update_gamma_H(value):
    global gamma_H
    gamma_H = value / 100
    apply_filters()

def update_c(value):
    global c
    c = value
    apply_filters()

def gaussian_high_pass_filter ( dft_image , radius, c) :

    rows, cols = dft_image.shape
    crow, ccol = rows // 2, cols // 2
    gaussian_filter = np.zeros((rows, cols), np.float32)

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            gaussian_filter[i, j] = np.exp(-( c *distance ** 2) / (2 * radius ** 2))

    return gaussian_filter

def homorphic_filter ( dft_image , sigma , gamma_L , gamma_H , c ) :

    return (gamma_H - gamma_L) * (1 - gaussian_high_pass_filter(dft_image, sigma, c)) + gamma_L

def apply_filters ():

    org_image = cv2.imread('gate_image.jpg',0)

    org_image = org_image.astype(np.uint8)

    log_image = log_of_image(org_image)
    dft_image = DFT_of_image(log_image)

    filtered_image = dft_image[0] * homorphic_filter(dft_image[0], radius, gamma_L, gamma_H, c)

    reconstructed_fft_shifted = filtered_image * np.exp(1j * dft_image[1])
    reconstructed_fft = np.fft.ifftshift(reconstructed_fft_shifted)
    image_reconstructed = np.abs(np.fft.ifft2(reconstructed_fft))
    image_reconstructed = cv2.normalize(image_reconstructed, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imshow("Output Image", image_reconstructed)
    cv2.imshow("Original Image", org_image)
    cv2.imshow("Filter",filtered_image)

radius = 1
gamma_L = 0.4
gamma_H = 3
c = 5

cv2.namedWindow("Filter")
cv2.createTrackbar("Radius", "Filter", 1,100, update_radius)
cv2.createTrackbar("gamma_L", "Filter", 1,100, update_gamma_L)
cv2.createTrackbar("gamma_H", "Filter", 1,500, update_gamma_H)
cv2.createTrackbar("c", "Filter", 1,100, update_c)
apply_filters()

cv2.waitKey(0)
cv2.destroyAllWindows()
