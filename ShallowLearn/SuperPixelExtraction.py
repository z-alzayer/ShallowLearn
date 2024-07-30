from skimage.filters import sobel, threshold_multiotsu
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.transform import resize
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, FastICA, LatentDirichletAllocation
from fast_slic.avx2 import SlicAvx2
import ShallowLearn.ImageHelper as ih
import matplotlib.pyplot as plt 

def felzenszwalb_segmentation(image, scale=100, sigma=0.5, min_size=50):
    segments_fz = felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
    return segments_fz

def slic_segmentation(image, n_segments=300, compactness=10, sigma=1, faster_slic = False):
    # segments_slic = Slic(image, num_components=n_segments, compactness=compactness, sigma=sigma, start_label=1)
    if faster_slic:
        slic_fast = SlicAvx2(num_components=n_segments, convert_to_lab=False, 
                             compactness=compactness, min_size_factor=0.25, subsample_stride=10)
        segments_slic = slic_fast.iterate(image)
    else:
        segments_slic = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=1)

    return segments_slic

def quickshift_segmentation(image, kernel_size=9, max_dist=30, ratio=0.5):
    segments_quick = quickshift(image, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
    return segments_quick

def watershed_segmentation(image, markers = 250, compactness=0.001):
    segments_watershed = watershed(image, markers, compactness=compactness)
    return segments_watershed

def multiotsu_thresholding(image, classes=3):
    thresholds = threshold_multiotsu(image, classes=classes)
    return thresholds

def pad_slice_segments(image, segments, shape = (32, 32, 13)):
    patches = []
    for i in np.unique(segments):
        segment = segments == i
        # if np.cumsum(segment) == 0:
        #     raise ValueError("Segment is empty")
        patch = image[segment]
        resized_patch = resize(patch, shape, preserve_range=True)
        patches.append(resized_patch)
    return np.array(patches) 


def pca_segments(patches, n_components=2):
    # pca = LatentDirichletAllocation(n_components=n_components)
    pca = KernelPCA(n_components=n_components)
    patches = patches.reshape(patches.shape[0], -1)
    pca.fit(patches)
    pca_image = pca.transform(patches)
    return pca_image

def optics_labels(image, segments, min_samples=100):
    from sklearn.cluster import OPTICS
    patches = pad_slice_segments(image, segments)
    pca_image = pca_segments(patches)
    # plt.scatter(pca_image[:,0], pca_image[:,1])
    # plt.show()
    db = OPTICS(min_samples=min_samples).fit(pca_image)
    return db.labels_


def generate_sup_pixel_labels(image):
    segments = slic_segmentation(image, n_segments = max(image.shape))
    labels = optics_labels(image, segments)
    super_pixel = np.zeros(segments.shape)
    
    for index, i in enumerate(np.unique(segments)):
        super_pixel[segments == i] = labels[index]
    return super_pixel


def gen_dii(image):
    super_pixel = generate_sup_pixel_labels(image)
    print(np.unique(super_pixel))
    mean_1 = np.mean(ih.apply_mask(image, np.expand_dims(super_pixel == 1, axis = 2))[:,:,3])
    mean_2 = np.mean(ih.apply_mask(image, np.expand_dims(super_pixel == 0, axis = 2))[:,:,3])
    if mean_1 < mean_2:
        mask_shallow = super_pixel == 0
        mask_deep = super_pixel == 1
    else:
        mask_shallow = super_pixel == 1
        mask_deep = super_pixel == 0
    dii_results = np.empty_like(image, dtype=np.float32)

    for band in range(13):
        # Extract band data
        band_data = image[:,:,band]
        
        # Calculate average reflectance in deep and shallow areas for the current band
        try:
            R_deep_avg = np.percentile(band_data[mask_deep],1)
        except:
            print("Unable to get a percentile")
            return None
            R_deep_avg = np.mean(band_data[mask_deep])
        # print(R_deep_avg)
        R_shallow_avg = np.mean(band_data[mask_shallow])        
        # Compute DII for each pixel in the current band
        # Use np.where to avoid taking the log of non-positive values
        R_diff = band_data - R_deep_avg
        dii_band = np.where(R_diff > 0, np.log(np.maximum(R_diff, 1e-10)), 0)
        
        # Store the DII result for the current band
        dii_results[:,:,band] = dii_band
        # dii_results = ih.apply_mask(dii_results, np.expand_dims(mask_deep, axis=2))
    return dii_results

def ostu_filter(dii_image):
    from skimage.filters import threshold_multiotsu

    thresholds = threshold_multiotsu(dii_image[:,:,[2]])
    regions = np.digitize(dii_image[:,:,[2]], bins=thresholds)
    return regions == 2