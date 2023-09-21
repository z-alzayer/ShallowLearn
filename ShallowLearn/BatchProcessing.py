import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import ShallowLearn.ImageHelper as ih
import ShallowLearn.RadiometricNormalisation as rn
import ShallowLearn.Transform as tf




def reshape_dataframe(df):
    """
    Reshape a DataFrame such that each unique 4-digit ID at the beginning of the strings
    becomes its own column, using the string following the ID as the column name.

    Parameters:
    - df: Input DataFrame with only one column of interest

    Returns:
    - Reshaped DataFrame
    """
    # Ensure the DataFrame has only one column of interest
    if df.shape[1] != 1:
        raise ValueError("Input DataFrame should have only one column of interest.")
    
    column_name = df.columns[0]
    
    # Extract the 4-digit identifier and the associated value
    df['ID'] = df[column_name].str.extract(r'(\d{4})')
    df['ColumnName'] = df[column_name].str[5:]
    
    # Set multi-index and then unstack to reshape the DataFrame
    df_reshaped = df.set_index(['ID', 'ColumnName']).drop(columns=column_name).unstack()
    
    # Drop top level of multi-index in columns and fill NaNs
    df_reshaped.columns = df_reshaped.columns.droplevel(0)
    df_reshaped.fillna('', inplace=True)

    return df_reshaped.reset_index(drop=True)


def extract_date_from_string(df, column_name='Image_name'):
    """
    Extract the date from the specified column string and set it to a separate date column.

    Parameters:
    - df: Input DataFrame
    - column_name: Name of the column containing strings with dates

    Returns:
    - DataFrame with an added 'Date' column
    """
    # Extract date using regex
    df['Date'] = df[column_name].str.extract(r'_(\d{8})T')
    
    # Convert the extracted date string to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    
    return df

def data_loader(img_no):
    img, meta, bounds = ih.load_img(path + data_frame.Path[img_no], return_meta = True)
    return img, meta, bounds


if __name__ == "__main__":

    path = "/media/ziad/Expansion/Cleaned_Data_Directory/"
    data_paths = os.listdir(path)
    data_paths = [i for i in data_paths if i.endswith(".tiff")]
    data_frame = pd.DataFrame(data_paths)
    reshape_dataframe(data_frame)
    data_frame.columns = ["Path", "ID", "Image_name"]
    extract_date_from_string(data_frame)
    
    ## They all work 
    #img, meta, bounds  = data_loader(3500)
    #rgb_img = ih.plot_rgb(img)
    #rgb_img = rgb_img.astype(np.float32)
    #rgb_img = np.where(rgb_img == 0, np.nan, rgb_img)


    #ih.plot_geotiff(rgb_img/255, bounds, data_frame.Path[3500])


    bounds_dict = {}
    for unique_id in data_frame.ID.unique():
        image_list = []
        date_list = []
        subset_df = data_frame[data_frame.ID == unique_id]
        for id, image, date in zip(subset_df.ID, subset_df.Path, subset_df.Date): 
            image_path = path + image
            # print(image_path)
            img, meta, bounds = ih.load_img(image_path, return_meta = True)
            print(img.shape)
            # ih.plot_rgb(img, plot=True)
            if img.shape[-1] != 13:
                continue
            date_list.append(date)
            image_list.append(img)

        corrected_images = []
        image_arr = np.array(image_list)
        ref = image_arr.mean(axis = 0)
        final_dates = []
        for index, image in enumerate(image_list):
            try:
                hist_norm = rn.pca_based_normalization(image,ref )    
                corrected_images.append(hist_norm)
                final_dates.append(date_list[index])
            except:
                print("Error in image {}".format(index))
        corrected_images = np.array(corrected_images)
        rgb_image = corrected_images[:, :, :, [3,2,1]]
        print(rgb_image.mean(axis=0).shape)
        # plt.hist(tf.LCE_multi(image_list.mean(axis = 0)).flatten(), bins = 100)
        plt.imshow(tf.LCE_multi(rgb_image.mean(axis = 0))/255)
        plt.savefig("/media/ziad/Expansion/Cleaned_Data_Directory/Corrected_Images/Mean_dataset_{}.png".format(unique_id))
        np.save("/media/ziad/Expansion/Cleaned_Data_Directory/Corrected_Images/Uncorrected_Images_{}.npy".format(unique_id), image_arr)
        np.save("/media/ziad/Expansion/Cleaned_Data_Directory/Corrected_Images/Corrected_Images_{}.npy".format(unique_id), corrected_images)
        np.save("/media/ziad/Expansion/Cleaned_Data_Directory/Corrected_Images/Corrected_Images_Dates_{}.npy".format(unique_id), final_dates)
        # concat bounds to df
        left,bottom, right, top = bounds
        bounds_dict[unique_id] = {"Left": left, "Bottom": bottom, "Right": right, "Top": top}
    bounds_df = pd.DataFrame(bounds_dict).T.to_csv("/media/ziad/Expansion/Cleaned_Data_Directory/Corrected_Images/bounds.csv")