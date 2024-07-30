from sklearn.decomposition import PCA
from sklearn.cluster import dbscan
import ShallowLearn.ImageHelper as ih
import ShallowLearn.FileProcessing as fp
from tqdm import tqdm
from osgeo import gdal



def get_file_list_from_directory(directory, extension = ".tiff", additional_filter = None):
    paths = fp.list_files_in_dir_recur(directory)
    paths = [path for path in paths if path.endswith(extension)]
    if additional_filter is None:
        return paths
    return [path for path in paths if (additional_filter) in path]



def generate_image_arr(paths):
    images = map(ih.load_img, paths)
    return images

def pca_and_cluster(image_list):


    images = []
    for image in image_list:
        print(image)
        break

    # # compute with 95% of the variance
    # pca_model = PCA(n_components=0.95)

    # cluster_model = dbscan()







if __name__ == "__main__":
    print("Hello world!")
    # path_to_fmask = "/media/zba21/Expansion/Cloud_Masks/"
    path = "/mnt/sda_mount/Clipped/L1C/"
# images = fp.list_files_in_dir_recur(path)
# single_reef = [i for i in images if "/34_" in i]
    paths = get_file_list_from_directory(path, ".tiff", "/34_")
    stack = []
    for i in tqdm(generate_image_arr(paths), desc="Processing files"):
        stack.append(i)
    print(len(stack))
        # break
    
    # pca_and_cluster(paths)