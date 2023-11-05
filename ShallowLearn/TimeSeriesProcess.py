import ShallowLearn.RadiometricNormalisation as rn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ShallowLearn.ImageHelper as ih

class ProcessImageArray():

    def __init__(self, arr, ref_idx = None):
        self.arr = arr
        
        if ref_idx is None:
            self.ref_img = self.arr.mean(axis=0)
        else:
            self.ref_img = self.arr[ref_idx]
        
    def apply_radiometric_norm(self):
        normalized_tensors = map(lambda src_img: rn.pca_based_normalization(src_img, self.ref_img), self.arr)
        
        # Convert the result back to a 4D numpy array
        normalized_4d = np.array(list(normalized_tensors))
        
        return normalized_4d

    def get_norm_imagery(self):
        return self.apply_radiometric_norm(self.arr)

def plot_image_from_path(path):
    img = ih.load_img(path, return_meta=True)
    return img
    

if __name__ == "__main__":
    data_frame = pd.read_csv("Data/merged_with_paths_to_data_df.csv")
    l1c_arr = []
    meta_arr = []
    bounds_arr = []
    for i in range(70):
        l1c = plot_image_from_path(data_frame[data_frame.Index == 2].FilePath_y.values[i])
        l1c_arr.append(l1c[0])
        meta_arr.append(l1c[1])
        bounds_arr.append(l1c[2])
# plot_image_from_path(data_frame[data_frame.Index == 2].FilePath_x.values[0])
    img_arr = np.array(l1c_arr)

    norm_img_arr = ProcessImageArray(img_arr)
    print(norm_img_arr.apply_radiometric_norm().shape)