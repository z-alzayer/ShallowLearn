import numpy as np

@dataclass
class SatelliteImage:
    """
    Class representing a satellite image
    """

    name : str
    path : str
    image : np.ndarray
    metadata : dict
    bounds : dict


    