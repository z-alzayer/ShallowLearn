import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import ShallowLearn.ImageHelper as ih

def create_gif_from_images(image_paths, output_gif_path, filter_string=".tiff", corrected_npy_suffix="Corrected_Images.npy", gif_duration=0.5):
    # Filter out the desired images based on the filter_string
    desired_images = [i for i in image_paths if filter_string in i]
    
    temp_folder = os.path.join(os.path.dirname(output_gif_path), "temp_gif_images")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    for idx, image_path in enumerate(desired_images):
        plt.imshow(ih.plot_rgb(ih.load_img(image_path)))
        plt.savefig(os.path.join(temp_folder, "{}.png".format(idx)))
    
    corrected_images_path = next((img for img in image_paths if corrected_npy_suffix in img), None)
    if corrected_images_path:
        corrected_images = np.load(corrected_images_path)

        for idx, img in enumerate(corrected_images):
            plt.imshow(ih.plot_rgb(img))
            plt.savefig(os.path.join(temp_folder, "{}_corrected.png".format(idx)))

    file_names = os.listdir(temp_folder)
    uncorrected_images = [i for i in file_names if i.endswith(".png") and not i.endswith("_corrected.png")]
    corrected_images = [i for i in file_names if i.endswith("_corrected.png")]

    with imageio.get_writer(output_gif_path, mode='I', duration=gif_duration) as writer:
        for filename in corrected_images:
            print(filename)
            image = imageio.imread(os.path.join(temp_folder, filename))
            writer.append_data(image)
    
    # Clean up the temporary files created
    for file_name in file_names:
        os.remove(os.path.join(temp_folder, file_name))
    os.rmdir(temp_folder)

def create_gif(directory_path, output_filepath):
    # Get all files in the directory
    file_list = os.listdir(directory_path)
    
    # Filter out non-PNG files
    png_files = [f for f in file_list if f.endswith('.png')]
    
    # Sort files by modification date
    png_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory_path, x)))
    
    # Create a list to hold the images
    images = []
    
    # Open each image and append it to the list
    for file in png_files:
        img_path = os.path.join(directory_path, file)
        images.append(Image.open(img_path))
    
    # Save the images as a GIF
    images[0].save(output_filepath,
                   save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)
    
    print(f"GIF saved at: {output_filepath}")

create_gif('/home/zba21/Documents/ShallowLearn/Animations/', '/home/zba21/Documents/ShallowLearn/Animations/prediction_reef_patch_34.gif')

# path = "/media/zba21/Expansion/Cleaned_Data_Directory"
# image_paths = fp.list_files_in_dir("/media/zba21/Expansion/Cleaned_Data_Directory")
# lizard_islands = [i for i in image_paths if "6880" in i and i.endswith(".tiff")]
# for i in range(len(lizard_islands)):
#     plt.imshow(ih.plot_rgb(ih.load_img(os.path.join(path, lizard_islands[i]))))
#     plt.savefig("../Graphs/Histogram_adjusted_gif/{}.png".format(i))
# corrected_lizard_island = np.load(os.path.join(path, "Corrected_Images", "Corrected_Images_6880.npy"))
# for i in range(len(lizard_islands)):
#     plt.imshow(ih.plot_rgb(corrected_lizard_island[i]))
#     # plt.show()
#     # if i == 5:
#     #     break
#     plt.savefig("../Graphs/Histogram_adjusted_gif/{}_corrected.png".format(i))
# file_names = os.listdir("../Graphs/Histogram_adjusted_gif")
# uncorrected_images = [i for i in file_names if i.endswith(".png") and not i.endswith("_corrected.png")]
# corrected_images = [i for i in file_names if i.endswith("_corrected.png")]
# with imageio.get_writer('../Graphs/corrected_lizard_island.gif', mode='I', duration = 0.5) as writer:
#     for filename in corrected_images:
#         print(filename)
#         image = imageio.imread(os.path.join("../Graphs/Histogram_adjusted_gif/" ,  filename))
#         writer.append_data(image)