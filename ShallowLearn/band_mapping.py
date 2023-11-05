band_mapping = {
    'B01': {'name': 'Coastal Aerosol', 'index': 0},
    'B02': {'name': 'Blue', 'index': 1},
    'B03': {'name': 'Green', 'index': 2},
    'B04': {'name': 'Red', 'index': 3},
    'B05': {'name': 'Red Edge 1', 'index': 4},
    'B06': {'name': 'Red Edge 2', 'index': 5},
    'B07': {'name': 'Red Edge 3', 'index': 6},
    'B08': {'name': 'NIR', 'index': 7},
    'B8A': {'name': 'Red Edge 4', 'index': 8},
    'B09': {'name': 'Water Vapor', 'index': 9},
    'B10': {'name': 'SWIR - Cirrus', 'index': 10},
    'B11': {'name': 'SWIR 1', 'index': 11},
    'B12': {'name': 'SWIR 2', 'index': 12}
}

geomorphic_dict = {
    0: {"color": "#000000", "description": "Unmapped"},
    11: {"color": "#77d0fc", "description": "Shallow Lagoon - Shallow Lagoon is any closed to semi-enclosed, sheltered, flat-bottomed shallow sediment-dominated lagoon area."},
    12: {"color": "#2ca2f9", "description": "Deep Lagoon - Deep Lagoon is any sheltered broad body of water semi-enclosed to enclosed by reef, with a variable depth (but shallower than surrounding ocean) and a soft bottom dominated by reef-derived sediment."},
    13: {"color": "#c5a7cb", "description": "Inner Reef Flat - Inner Reef Flat is a low energy, sediment-dominated, horizontal to gently sloping platform behind the Outer Reef Flat."},
    14: {"color": "#92739d", "description": "Outer Reef Flat - Adjacent to the seaward edge of the reef, Outer Reef Flat is a level (near horizontal), broad and shallow platform that displays strong wave-driven zonation"},
    15: {"color": "#614272", "description": "Reef Crest - Reef Crest is a zone marking the boundary between the reef flat and the reef slope, generally shallow and characterized by highest wave energy absorbance."},
    16: {"color": "#fbdefb", "description": "Terrestrial Reef Flat - Terrestrial Reef Flat is a broad, flat, shallow to semi-exposed area of fringing reef found directly attached to land at one side, and subject to freshwater run-off, nutrients and sediment."},
    21: {"color": "#10bda6", "description": "Sheltered Reef Slope - Sheltered Reef Slope is any submerged, sloping area extending into Deep Water but protected from strong directional prevailing wind or current, either by land or by opposing reef structures."},
    22: {"color": "#288471", "description": "Reef Slope - Reef Slope is a submerged, sloping area extending seaward from the Reef Crest (or Flat) towards the shelf break. Windward facing, or any direction if no dominant prevailing wind or current exists."},
    23: {"color": "#cd6812", "description": "Plateau - Plateau is any deeper submerged, hard-bottomed, horizontal to gently sloping seaward facing reef feature."},
    24: {"color": "#befbff", "description": "Back Reef Slope - Back Reef Slope is a complex, interior, - often gently sloping - reef zone occurring behind the Reef Flat. Of variable depth (but deeper than Reef Flat and more sloped), it is sheltered, sediment-dominated and often punctuated by coral outcrops."},
    25: {"color": "#ffba15", "description": "Patch Reef - Patch Reef is any small, detached to semi-detached lagoonal coral outcrop arising from sand-bottomed area."}
}

benthic_dict = {
    0: {"color": "#000000", "description": "Unmapped"},
    11: {"color": "#ffffbe", "description": "Sand - Sand is any soft-bottom area dominated by fine unconsolidated sediments."},
    12: {"color": "#e0d05e", "description": "Rubble - Rubble is any habitat featuring loose, rough fragments of broken reef material."},
    13: {"color": "#b19c3a", "description": "Rock - Rock is any exposed area of hard bare substrate."},
    14: {"color": "#668438", "description": "Seagrass - Seagrass is any habitat where seagrass is the dominant biota."},
    15: {"color": "#ff6161", "description": "Coral/Algae - Coral/Algae is any hard-bottom area supporting living coral and/or algae."},
    18: {"color": "#9bcc4f", "description": "Microalgal Mats - Microalgal Mats are any visible accumulations of microscopic algae in sandy sediments."}
}

reef_mask_dict = {
    0: {"color": "#000000", "description": "Not reef"},
    1: {"color": "#ffffff", "description": "Reef"}
}