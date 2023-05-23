import unittest
import numpy as np
from ShallowLearn.band_mapping import band_mapping
from ShallowLearn import Indices as band_indices

class TestBandIndices(unittest.TestCase):

    def setUp(self):
        self.image = np.random.rand(10, 10, 13)  # Adjust the shape based on your data

    def test_ci(self):
        ci = band_indices.ci(self.image)
        self.assertEqual(ci.shape, (10, 10))
        # Add more assertions to verify the correctness of CI computation

    def test_oci(self):
        oci = band_indices.oci(self.image)
        self.assertEqual(oci.shape, (10, 10))
        # Add more assertions to verify the correctness of OCI computation

    def test_ssi(self):
        ssi = band_indices.ssi(self.image)
        self.assertEqual(ssi.shape, (10, 10))
        # Add more assertions to verify the correctness of SSI computation

    def test_ti(self):
        ti = band_indices.ti(self.image)
        self.assertEqual(ti.shape, (10, 10))
        # Add more assertions to verify the correctness of TI computation

    def test_wqi(self):
        wqi = band_indices.wqi(self.image)
        self.assertEqual(wqi.shape, (10, 10))
        # Add more assertions to verify the correctness of WQI computation

    def test_ndci(self):
        ndci = band_indices.ndci(self.image)
        self.assertEqual(ndci.shape, (10, 10))
        # Add more assertions to verify the correctness of NDCI computation

if __name__ == '__main__':
    unittest.main()
