sentinel_2_satellite_data = {
    'Band Name': ["Coastal aerosol", "Blue", "Green", "Red", "Vegetation Red Edge", "Vegetation Red Edge", "Vegetation Red Edge", "NIR", "Narrow NIR", "Water vapour", "SWIR – Cirrus", "SWIR", "SWIR"],
    'Sensor': ["MSI"]*13,
    'Band Number': [i for i in range(1, 14)],
    'Sentinel-2A Wavelength': [443.9, 496.6, 560.0, 664.5, 703.9, 740.2, 782.5, 835.1, 864.8, 945.0, 1373.5, 1613.7, 2202.4],
    'Bandwidth A': [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180],
    'Sentinel-2B Wavelength': [442.3, 492.1, 559, 665, 703.8, 739.1, 779.7, 833, 864, 943.2, 1376.9, 1610.4, 2185.7],
    'Bandwidth B': [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180],
    'Resolution': [60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 60, 20, 20]
}

import matplotlib.pyplot as plt
import numpy as np

# Gaussian-like curve function
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# Using the provided 'data' dictionary
fig, ax = plt.subplots(figsize=(12, 8))

x = np.linspace(400, 2300, 5000)  # x-axis range

# Generate a color palette
colors = plt.cm.viridis(np.linspace(0, 1, len(sentinel_2_satellite_data['Band Name'])))

# Plotting Sentinel-2A sentinel_2_satellite_data
for idx, (band, wavelength, bandwidth) in enumerate(zip(sentinel_2_satellite_data['Band Name'], sentinel_2_satellite_data['Sentinel-2A Wavelength'], sentinel_2_satellite_data['Bandwidth A'])):
    y = gaussian(x, wavelength, bandwidth/2.355)  # 2.355 is a factor to approximate FWHM for Gaussian
    ax.fill_between(x, y, where=(y>0.01), color=colors[idx], alpha=0.6)  # Only fill where curve is above a threshold
    # ax.plot(wavelength, 0.5, 'o', color=colors[idx], label=f'Sentinel-2A {band}')

# Plotting Sentinel-2B sentinel_2_satellite_data
for idx, (band, wavelength, bandwidth) in enumerate(zip(sentinel_2_satellite_data['Band Name'], sentinel_2_satellite_data['Sentinel-2B Wavelength'], sentinel_2_satellite_data['Bandwidth B'])):
    y = gaussian(x, wavelength, bandwidth/2.355)
    ax.fill_between(x, y, where=(y>0.01), color=colors[idx], alpha=0.6)
    # ax.plot(wavelength, 0.5, 'x', color=colors[idx], label=f'Sentinel-2B {band}')

# Aesthetics
ax.set_title("Sentinel-2 Satellite Wavelengths")
ax.set_xlabel("Wavelength (nm)")
ax.set_yticks([])
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))

# Display the plot
plt.tight_layout()
plt.savefig("Graphs/Sentinel-2 Wavelengths.png")
