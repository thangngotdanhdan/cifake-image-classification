import cv2
import numpy as np
import joblib
from scipy.fftpack import dct
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
import pywt

def scipy_dct2d(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def preprocess_single_image(image_path, size=(32, 32)):
    # 1. Read Image in both grayscale and color
    img_color = cv2.imread(str(image_path))
    if img_color is None:
        raise ValueError(f"Failed to read image at: {image_path}")

    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Resize images to standard 32x32 size
    img_gray_resized = cv2.resize(img_gray, size)
    img_color_resized = cv2.resize(img_color, size)

    # 1. Grayscale DCT Features (100 coefficients)
    dct_coefs = scipy_dct2d(img_gray_resized.astype(np.float32))
    num_coefficients = 100
    k = int(np.sqrt(num_coefficients))
    dct_feats = dct_coefs[:k, :k].flatten()

    # 2. HOG Features (324 features)
    hog_feats = hog(
        img_gray_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    # 3. LBP Features (10 features)
    P, R = 8, 1
    lbp = local_binary_pattern(img_gray_resized, P, R, method='uniform')
    n_bins = 10
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # 4. GLCM Features (16 features)
    glcm = graycomatrix(img_gray_resized, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    glcm_feats = np.concatenate([contrast, correlation, energy, homogeneity])

    # 5. Wavelet Features (8 features)
    coeffs = pywt.dwt2(img_gray_resized, 'db1')
    cA, (cH, cV, cD) = coeffs
    wavelet_feats = np.array([
        np.mean(cA), np.std(cA),
        np.mean(cH), np.std(cH),
        np.mean(cV), np.std(cV),
        np.mean(cD), np.std(cD)
    ])

    # 6. Color Histogram Features (512 features)
    bins = (8, 8, 8)
    color_hist = cv2.calcHist([img_color_resized], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(color_hist, color_hist)
    color_hist_feats = color_hist.flatten()

    # Concatenate all features to reconstruct the 970-dimensional vector
    # E9 combination order: (DCT + HOG + LBP) + GLCM + Wavelet + Color Histogram
    feature_vector = np.hstack((
        dct_feats,
        hog_feats,
        lbp_hist,
        glcm_feats,
        wavelet_feats,
        color_hist_feats
    ))

    # Reshape to (1, 970) for model input prediction
    return feature_vector.reshape(1, -1)
