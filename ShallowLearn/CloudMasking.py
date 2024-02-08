import ShallowLearn.ImageHelper as ih
import matplotlib.pyplot as plt
#%%
path = "/media/zba21/Expansion/Cloud_Masks/fmask_S2A_MSIL1C_20200829T003711_N0209_R059_T55LCD_20200829T020819.SAFE"
mask = ih.load_img(path)
fig, ax = plt.subplots(1,1, figsize=(20,20))
ax.imshow(mask)
# plt.colorbar()
plt.show()
#%%
