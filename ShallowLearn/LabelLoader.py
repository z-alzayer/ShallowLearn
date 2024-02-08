from ShallowLearn.LoadData import LoadGeoTIFF

class LabelLoader(LoadGeoTIFF):

    def __init__(self, data_source):
        super().__init__(data_source)
    
    def load(self):
        img = super().load()
        img = img.swapaxes(0,2)
        img = img.swapaxes(0,1)
        return img
    


if __name__ == "__main__":
    benthic_labels = "/mnt/sda_mount/Clipped/GBR_2017/benthic_2/44_benthic_2.tif"
    geomorphic_labels = "/mnt/sda_mount/Clipped/GBR_2017/geomorphic/44_geomorphic.tif"
    benthic_labels = LabelLoader(benthic_labels).load()
    geomorphic_labels = LabelLoader(geomorphic_labels).load()
    print(benthic_labels.shape)
    print(geomorphic_labels.shape)