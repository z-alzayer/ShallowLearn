import numpy as np
from ShallowLearn.band_mapping import band_mapping
import ShallowLearn.Indices as band_indices
import pytest

class TestBandIndices:
    @classmethod
    def setup_class(cls):
        cls.image = np.random.rand(10, 10, 13)  # Adjust the shape based on your data

    def test_ci(self):
        ci = band_indices.ci(self.image)
        assert ci.shape == (10, 10)
        # Add more assertions to verify the correctness of CI computation

    def test_oci(self):
        oci = band_indices.oci(self.image)
        assert oci.shape == (10, 10)
        # Add more assertions to verify the correctness of OCI computation

    def test_ssi(self):
        ssi = band_indices.ssi(self.image)
        assert ssi.shape == (10, 10)
        # Add more assertions to verify the correctness of SSI computation

    def test_ti(self):
        ti = band_indices.ti(self.image)
        assert ti.shape == (10, 10)
        # Add more assertions to verify the correctness of TI computation

    def test_wqi(self):
        wqi = band_indices.wqi(self.image)
        assert wqi.shape == (10, 10)
        # Add more assertions to verify the correctness of WQI computation

    def test_ndci(self):
        ndci = band_indices.ndci(self.image)
        assert ndci.shape == (10, 10)
        # Add more assertions to verify the correctness of NDCI computation
