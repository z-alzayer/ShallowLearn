"""
Test script for superpixel DII pipeline with fake data.
"""
import numpy as np
import matplotlib.pyplot as plt
from ShallowLearn.segmentation.superpixels import create_superpixel_dii_stack, process_superpixel_dii_pipeline

def create_fake_satellite_image(height=100, width=100, n_bands=10):
    """
    Create a fake satellite image with realistic spectral characteristics.
    
    Parameters:
    -----------
    height : int
        Image height
    width : int  
        Image width
    n_bands : int
        Number of spectral bands
    
    Returns:
    --------
    np.ndarray
        Fake satellite image (height, width, bands)
    """
    np.random.seed(42)  # For reproducible results
    
    # Create base image with different regions
    image = np.zeros((height, width, n_bands))
    
    # Create different spectral signatures for different regions
    # Deep water (lower reflectance, especially in NIR)
    deep_water_signature = np.array([0.02, 0.03, 0.04, 0.03, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002])
    
    # Shallow water (higher reflectance, varies by band)
    shallow_water_signature = np.array([0.05, 0.07, 0.09, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01])
    
    # Land/coral (much higher reflectance in NIR)
    land_signature = np.array([0.08, 0.12, 0.15, 0.18, 0.25, 0.35, 0.40, 0.45, 0.50, 0.52])
    
    # Create spatial regions
    # Deep water in center
    center_y, center_x = height // 2, width // 2
    
    for i in range(height):
        for j in range(width):
            # Distance from center
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            
            if dist < 20:  # Deep water region
                base_signature = deep_water_signature
                noise_scale = 0.005
            elif dist < 35:  # Shallow water region  
                base_signature = shallow_water_signature
                noise_scale = 0.01
            else:  # Land/coral region
                base_signature = land_signature
                noise_scale = 0.02
            
            # Add some noise to make it realistic
            noise = np.random.normal(0, noise_scale, n_bands)
            image[i, j, :] = np.maximum(0, base_signature + noise)
    
    return image.astype(np.float32)

def test_superpixel_dii_pipeline():
    """
    Test the superpixel DII pipeline with fake data.
    """
    print("Creating fake satellite image...")
    
    # Create fake satellite image
    image = create_fake_satellite_image(height=80, width=80, n_bands=10)
    
    print(f"Image shape: {image.shape}")
    print(f"Image value range: {image.min():.4f} to {image.max():.4f}")
    
    # Test the main pipeline function
    print("\nTesting create_superpixel_dii_stack...")
    
    try:
        features, segments, results = create_superpixel_dii_stack(
            image,
            n_segments=50,  # Fewer segments for small test image
            bands=[0, 1, 2],
            correction_factor=10
        )
        
        print("✓ create_superpixel_dii_stack completed successfully")
        print(f"Features shape: {features.shape}")
        print(f"Segments shape: {segments.shape}")
        print(f"Unique segments: {len(np.unique(segments))}")
        
        # Check results dictionary
        expected_keys = ['segments', 'cluster_map', 'deep_mask', 'shallow_mask', 
                        'deep_idx', 'cluster_labels', 'features', 'transformed_features',
                        'gmm', 'pca', 'dii_stack', 'band_combos']
        
        for key in expected_keys:
            if key in results:
                print(f"✓ Results contains '{key}'")
            else:
                print(f"✗ Results missing '{key}'")
        
        # Check DII stack
        dii_stack = results['dii_stack']
        print(f"DII stack shape: {dii_stack.shape}")
        print(f"DII stack value range: {dii_stack.min():.4f} to {dii_stack.max():.4f}")
        
        # Check masks
        deep_mask = results['deep_mask']
        shallow_mask = results['shallow_mask']
        print(f"Deep pixels: {np.sum(deep_mask)}")
        print(f"Shallow pixels: {np.sum(shallow_mask)}")
        print(f"Deep cluster index: {results['deep_idx']}")
        
        # Verify band combinations were used
        band_combos = results['band_combos']
        print(f"Number of band combinations: {len(band_combos)}")
        print(f"Band combinations used: {band_combos[:5]}...")  # Show first 5
        
        return True, results
        
    except Exception as e:
        print(f"✗ Error in create_superpixel_dii_stack: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def test_direct_pipeline():
    """
    Test the direct pipeline function.
    """
    print("\n" + "="*50)
    print("Testing process_superpixel_dii_pipeline directly...")
    
    # Create fake data
    image = create_fake_satellite_image(height=60, width=60, n_bands=10)
    
    # Create fake segments (simple grid)
    segments = np.zeros((60, 60), dtype=int)
    segment_size = 10
    segment_id = 1
    
    for i in range(0, 60, segment_size):
        for j in range(0, 60, segment_size):
            segments[i:i+segment_size, j:j+segment_size] = segment_id
            segment_id += 1
    
    print(f"Created {np.max(segments)} segments")
    
    try:
        from ShallowLearn.segmentation.superpixels import process_superpixel_dii_pipeline
        
        results = process_superpixel_dii_pipeline(
            image, 
            segments,
            bands=[0, 1, 2],
            n_components=3
        )
        
        print("✓ process_superpixel_dii_pipeline completed successfully")
        
        # Test with custom band combinations
        custom_bands = [(0, 1), (1, 2), (2, 3)]
        results_custom = process_superpixel_dii_pipeline(
            image,
            segments, 
            bands=[0, 1, 2],
            band_combos=custom_bands
        )
        
        print("✓ Custom band combinations work")
        print(f"Custom DII stack shape: {results_custom['dii_stack'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in process_superpixel_dii_pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def visualize_results(image, results):
    """
    Create visualizations of the results.
    """
    print("\nCreating visualizations...")
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image (RGB)
        rgb_image = image[:, :, [2, 1, 0]]  # Assuming BGR order
        rgb_image = np.clip(rgb_image / rgb_image.max(), 0, 1)
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title('Original Image (RGB)')
        axes[0, 0].axis('off')
        
        # Segments
        axes[0, 1].imshow(results['segments'], cmap='nipy_spectral')
        axes[0, 1].set_title('Superpixel Segments')
        axes[0, 1].axis('off')
        
        # Cluster map
        axes[0, 2].imshow(results['cluster_map'], cmap='viridis')
        axes[0, 2].set_title('Cluster Map')
        axes[0, 2].axis('off')
        
        # Deep mask
        axes[1, 0].imshow(results['deep_mask'], cmap='Blues')
        axes[1, 0].set_title('Deep Water Mask')
        axes[1, 0].axis('off')
        
        # Shallow mask
        axes[1, 1].imshow(results['shallow_mask'], cmap='Reds')
        axes[1, 1].set_title('Shallow Water Mask')
        axes[1, 1].axis('off')
        
        # DII example (first band combination)
        dii_example = results['dii_stack'][:, :, 0]
        im = axes[1, 2].imshow(dii_example, cmap='RdYlBu_r')
        axes[1, 2].set_title('DII Example (Band Combo 0)')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('/Users/ziad/Documents/GitHub/ShallowLearn/test_superpixel_results.png', 
                   dpi=150, bbox_inches='tight')
        print("✓ Visualization saved as 'test_superpixel_results.png'")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating visualizations: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Superpixel DII Pipeline")
    print("="*50)
    
    # Run main test
    success, results = test_superpixel_dii_pipeline()
    
    if success:
        print("\n✓ Main pipeline test PASSED")
        
        # Test direct pipeline
        direct_success = test_direct_pipeline()
        
        if direct_success:
            print("\n✓ Direct pipeline test PASSED")
            
            # Create visualizations
            image = create_fake_satellite_image(height=80, width=80, n_bands=10)
            visualize_results(image, results)
            
            print("\n" + "="*50)
            print("ALL TESTS PASSED! 🎉")
            print("The superpixel DII pipeline is working correctly.")
        else:
            print("\n✗ Direct pipeline test FAILED")
    else:
        print("\n✗ Main pipeline test FAILED")