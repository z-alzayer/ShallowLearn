import numpy as np
import pytest
import ShallowLearn.Transform as trf

class TestImageProcessing:
    @classmethod
    def setup_class(cls):
        # Create test images with different channel configurations
        cls.rgb_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cls.multiband_image = np.random.rand(64, 64, 5) * 255  # 5-channel image
        cls.gray_image = np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)

    def test_scalers(self):
        # Test standard scaler
        scaled_std = trf.mutliband_standard_scaler(self.multiband_image)
        assert scaled_std.shape == self.multiband_image.shape
        
        # Test robust scaler
        scaled_rob = trf.mutliband_robust_scaler(self.multiband_image)
        assert scaled_rob.shape == self.multiband_image.shape

    def test_BCET(self):
        # Single channel
        enhanced = trf.BCET(self.gray_image[..., 0])
        assert enhanced.min() >= 0
        assert enhanced.max() <= 255
        
        # Multi-channel
        enhanced_multi = trf.BCET_multi(self.rgb_image)
        assert enhanced_multi.shape == self.rgb_image.shape

    def test_contrast_enhancement(self):
        # Test LCE
        lce = trf.linear_contrast_enhancement(self.gray_image[..., 0])
        assert lce.shape == self.gray_image.shape[:2]
        
        # Test LCE multi
        lce_multi = trf.LCE_multi(self.rgb_image)
        assert lce_multi.shape == self.rgb_image.shape

    def test_color_space_conversion(self):
        # Test RGB to HSI and back
        hsi = trf.rgb_to_hsi(self.rgb_image)
        assert hsi.shape == self.rgb_image.shape
        
        rgb = trf.hsi_to_rgb(hsi)
        assert rgb.shape == self.rgb_image.shape
    # These need reworking of how they currently work - they work with multiband images
    # def test_lab_transforms(self):
    #     # Test LAB stretching
    #     lab = trf.transform_lab_stretch(self.rgb_image)
    #     assert lab.shape == self.rgb_image.shape
        
    #     # Test multiband LAB
    #     lab_multi = trf.transform_multiband_lab(self.multiband_image)
    #     assert lab_multi.shape == self.multiband_image.shape

    # def test_hsv_transforms(self):
    #     # Test HSV stretching
    #     hsv = trf.transform_hsv_stretch(self.rgb_image)
    #     assert hsv.shape == self.rgb_image.shape
        
    #     # Test multiband HSV
    #     hsv_multi = trf.transform_multiband_hsv(self.multiband_image)
    #     assert hsv_multi.shape == self.multiband_image.shape

    def test_exceptions(self):
        # Test uniform image error
        uniform_image = np.full((10, 10), 100)
        with pytest.raises(ValueError):
            trf.linear_contrast_enhancement(uniform_image)
