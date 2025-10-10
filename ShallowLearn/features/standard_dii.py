import rasterio 
import rasterio.mask
import geopandas as gpd 
import numpy as np
from scipy import linalg
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import xgboost as xgb

def extract_raster_values(shapefile_path, raster_path, bands=None):
    """
    Extract raster values using shapefile geometries as masks.
    
    Args:
        shapefile_path: Path to the shapefile
        raster_path: Path to the raster file
        bands: Optional list of band indices to extract
    
    Returns:
        List of numpy arrays containing masked raster values for each geometry
    """
    # Read the shapefile using GeoPandas
    shapes = gpd.read_file(shapefile_path)
    
    # Read the raster using rasterio
    with rasterio.open(raster_path) as src:
        # Check if CRS match
        if shapes.crs != src.crs:
            # Reproject shapefile to match raster CRS
            shapes = shapes.to_crs(src.crs)
        
        # Create list to store values for each geometry
        values = []
        
        # Extract values for each geometry
        for geometry in shapes.geometry:
            # Create mask from geometry
            masked_data, _ = rasterio.mask.mask(src, [geometry], crop=True)
            
            # If specific bands requested, extract only those
            if bands is not None:
                masked_data = masked_data[bands, :, :]
            
            values.append(masked_data)
        
        return values


# ============================================================================
# CALCULATION FUNCTIONS (for training/calibration)
# ============================================================================

def calculate_depth_invariant_indices(deep_areas, shallow_areas, band_i_idx, band_j_idx, 
                                     method='linear', **method_kwargs):
    """
    Calculate depth invariant indices using various regression methods.
    
    Parameters:
    -----------
    deep_areas : list of arrays
        Deep water areas for each band
    shallow_areas : list of arrays
        Shallow water areas (same bottom type) for each band
    band_i_idx : int
        Index for band i
    band_j_idx : int
        Index for band j
    method : str
        Regression method: 'linear', 'polynomial', 'spline', 'neural_network', 'xgboost'
    **method_kwargs : dict
        Additional arguments for specific methods:
        - poly_degree (int): for 'polynomial' method (default: 2)
        - smoothing (float): for 'spline' method (default: None)
        - spline_k (int): spline degree (default: 3)
        - hidden_layers (tuple): for 'neural_network' method (default: (16, 8))
        - max_iter (int): for 'neural_network' method (default: 1000)
        - n_estimators (int): for 'xgboost' method (default: 100)
        - max_depth (int): for 'xgboost' method (default: 6)
        - learning_rate (float): for 'xgboost' method (default: 0.1)
    
    Returns:
    --------
    model : various types
        Fitted model (slope, coefficients, params, or spline object)
    Ls : tuple
        Deep water means (Ls_i, Ls_j)
    """
    # Calculate deep water means
    deep_i = np.concatenate([area[band_i_idx].flatten() for area in deep_areas])
    deep_j = np.concatenate([area[band_j_idx].flatten() for area in deep_areas])
    Ls_i = np.nanmean(deep_i)
    Ls_j = np.nanmean(deep_j)
    
    # Calculate regression for same bottom areas
    shallow_i = np.concatenate([area[band_i_idx].flatten() for area in shallow_areas])
    shallow_j = np.concatenate([area[band_j_idx].flatten() for area in shallow_areas])
    
    # Apply minimum difference threshold of 0.01
    dif_i = np.maximum(shallow_i - Ls_i, 0.01)
    dif_j = np.maximum(shallow_j - Ls_j, 0.01)
    
    # Transform to log space
    Xi = np.log(dif_i)
    Xj = np.log(dif_j)
    
    # Remove NaN/Inf values
    valid_mask = np.isfinite(Xi) & np.isfinite(Xj)
    Xi_clean = Xi[valid_mask]
    Xj_clean = Xj[valid_mask]
    
    # Fit model based on method
    if method == 'linear':
        model = _fit_linear(Xi_clean, Xj_clean)
    elif method == 'polynomial':
        poly_degree = method_kwargs.get('poly_degree', 2)
        model = _fit_polynomial(Xi_clean, Xj_clean, poly_degree)
    elif method == 'neural_network':
        hidden_layers = method_kwargs.get('hidden_layers', (16, 8))
        max_iter = method_kwargs.get('max_iter', 1000)
        model = _fit_neural_network(Xi_clean, Xj_clean, hidden_layers, max_iter)
    elif method == 'xgboost':
        n_estimators = method_kwargs.get('n_estimators', 100)
        max_depth = method_kwargs.get('max_depth', 6)
        learning_rate = method_kwargs.get('learning_rate', 0.1)
        # Pass any additional XGBoost parameters
        extra_kwargs = {k: v for k, v in method_kwargs.items() 
                       if k not in ['n_estimators', 'max_depth', 'learning_rate']}
        model = _fit_xgboost(Xi_clean, Xj_clean, n_estimators, max_depth, learning_rate, **extra_kwargs)
    elif method == 'spline':
        smoothing = method_kwargs.get('smoothing', None)
        spline_k = method_kwargs.get('spline_k', 3)
        model = _fit_spline(Xi_clean, Xj_clean, smoothing, spline_k)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'linear', 'polynomial', 'neural_network', 'xgboost', 'spline'")
    
    return model, (Ls_i, Ls_j)



def calculate_slope_from_values(deep_i, deep_j, shallow_i, shallow_j, 
                                method='linear', **method_kwargs):
    """
    Calculate slope/model from raw values using various regression methods.
    
    Parameters:
    -----------
    deep_i, deep_j : arrays
        Deep water values for bands i and j
    shallow_i, shallow_j : arrays
        Shallow water values for bands i and j
    method : str
        Regression method: 'linear', 'polynomial', 'spline', 'neural_network', 'xgboost'
    **method_kwargs : dict
        Additional arguments for specific methods
    
    Returns:
    --------
    model : various types
        Fitted model
    Ls : tuple
        Deep water means (Ls_i, Ls_j)
    """
    # Ls_i = np.nanmean(deep_i)
    # Ls_j = np.nanmean(deep_j)
    Ls_i = np.nanpercentile(deep_i, 50, method='lower')
    Ls_j = np.nanpercentile(deep_j, 50, method='lower')
    # Ls_i = np.nanmedian(deep_i)
    # Ls_j = np.nanmedian(deep_j)

    # Apply minimum difference threshold of 0.01
    dif_i = np.maximum(shallow_i - Ls_i, 0.01)
    dif_j = np.maximum(shallow_j - Ls_j, 0.01)
    
    # Transform to log space
    Xi = np.log(dif_i)
    Xj = np.log(dif_j)
    
    # Remove NaN/Inf values
    valid_mask = np.isfinite(Xi) & np.isfinite(Xj)
    Xi_clean = Xi[valid_mask]
    Xj_clean = Xj[valid_mask]
    
    # Fit model based on method
    if method == 'linear':
        model = _fit_linear(Xi_clean, Xj_clean)
    elif method == 'polynomial':
        poly_degree = method_kwargs.get('poly_degree', 2)
        model = _fit_polynomial(Xi_clean, Xj_clean, poly_degree)
    elif method == 'neural_network':
        hidden_layers = method_kwargs.get('hidden_layers', (16, 8))
        max_iter = method_kwargs.get('max_iter', 1000)
        model = _fit_neural_network(Xi_clean, Xj_clean, hidden_layers, max_iter)
    elif method == 'xgboost':
        n_estimators = method_kwargs.get('n_estimators', 100)
        max_depth = method_kwargs.get('max_depth', 6)
        learning_rate = method_kwargs.get('learning_rate', 0.1)
        extra_kwargs = {k: v for k, v in method_kwargs.items() 
                       if k not in ['n_estimators', 'max_depth', 'learning_rate']}
        model = _fit_xgboost(Xi_clean, Xj_clean, n_estimators, max_depth, learning_rate, **extra_kwargs)
    elif method == 'spline':
        smoothing = method_kwargs.get('smoothing', None)
        spline_k = method_kwargs.get('spline_k', 3)
        model = _fit_spline(Xi_clean, Xj_clean, smoothing, spline_k)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'linear', 'polynomial', 'neural_network', 'xgboost', 'spline'")
    
    return model, (Ls_i, Ls_j)

# ============================================================================
# FITTING FUNCTIONS (internal helpers)
# ============================================================================
def _fit_xgboost(Xi, Xj, n_estimators=100, max_depth=6, learning_rate=0.1, **xgb_kwargs):
    """Fit an XGBoost regressor"""
    
    Xi_reshaped = Xi.reshape(-1, 1)
    
    # Create XGBoost regressor
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='reg:squarederror',
        random_state=42,
        **xgb_kwargs
    )
    
    model.fit(Xi_reshaped, Xj)
    
    return {'model': model, 'model_type': 'xgboost'}
def _fit_linear(Xi, Xj):
    """Fit linear regression: Xj = slope * Xi + intercept"""
    slope, intercept = stats.linregress(Xi, Xj)[:2]
    return {'slope': slope, 'intercept': intercept}


def _fit_polynomial(Xi, Xj, degree=2):
    """Fit polynomial regression"""
    coeffs = np.polyfit(Xi, Xj, degree)
    return {'coeffs': coeffs, 'degree': degree}


def _fit_neural_network(Xi, Xj, hidden_layers=(16, 8), max_iter=1000):
    """Fit a neural network regressor"""
    from sklearn.neural_network import MLPRegressor
    
    Xi_reshaped = Xi.reshape(-1, 1)
    
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        alpha=0.001  # L2 regularization
    )
    
    mlp.fit(Xi_reshaped, Xj)
    
    return {'model': mlp, 'model_type': 'neural_network'}


def _fit_spline(Xi, Xj, smoothing=None, k=3):
    """Fit spline model"""
    sort_idx = np.argsort(Xi)
    Xi_sorted = Xi[sort_idx]
    Xj_sorted = Xj[sort_idx]
    
    spline = UnivariateSpline(Xi_sorted, Xj_sorted, s=smoothing, k=k)
    return {'spline': spline}


# ============================================================================
# PREDICTION FUNCTIONS (internal helpers)
# ============================================================================

def _predict_linear(log_i, model):
    """Predict using linear model"""
    return model['slope'] * log_i + model['intercept']


def _predict_polynomial(log_i, model):
    """Predict using polynomial model"""
    return np.polyval(model['coeffs'], log_i)


def _predict_neural_network(log_i, model):
    """Predict using neural network"""
    mlp = model['model']
    original_shape = log_i.shape
    log_i_reshaped = log_i.reshape(-1, 1)
    predictions = mlp.predict(log_i_reshaped)
    return predictions.reshape(original_shape)


def _predict_spline(log_i, model):
    """Predict using spline model"""
    return model['spline'](log_i)

def _predict_xgboost(log_i, model):
    """Predict using XGBoost"""
    xgb_model = model['model']
    original_shape = log_i.shape
    log_i_reshaped = log_i.reshape(-1, 1)
    predictions = xgb_model.predict(log_i_reshaped)
    return predictions.reshape(original_shape)

# ============================================================================
# APPLICATION FUNCTION (for inference on full images)
# ============================================================================

def apply_depth_invariant_index(image_i, image_j, model, Ls, method='linear'):
    """
    Apply the depth invariant index to full images using specified method.
    
    Parameters:
    -----------
    image_i : array
        Full image for band i
    image_j : array
        Full image for band j
    model : various types
        Fitted model from calculate_depth_invariant_indices
    Ls : tuple
        Deep water means (Ls_i, Ls_j)
    method : str
        Method used: 'linear', 'polynomial', 'neural_network', 'xgboost', 'spline'
        Must match the method used in calculate_depth_invariant_indices
    
    Returns:
    --------
    dii : array
        Depth invariant index image
    """
    Ls_i, Ls_j = Ls
    
    # Apply minimum difference threshold
    dif_i = np.maximum(image_i - Ls_i, 0.01)
    dif_j = np.maximum(image_j - Ls_j, 0.01)
    
    # Transform to log space
    log_i = np.log(dif_i)
    log_j = np.log(dif_j)
    
    # Predict log_j based on method
    if method == 'linear':
        log_j_pred = _predict_linear(log_i, model)
    elif method == 'polynomial':
        log_j_pred = _predict_polynomial(log_i, model)
    elif method == 'neural_network':
        log_j_pred = _predict_neural_network(log_i, model)
    elif method == 'xgboost':
        log_j_pred = _predict_xgboost(log_i, model)
    elif method == 'spline':
        log_j_pred = _predict_spline(log_i, model)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Return residual (depth invariant index)
    return log_j - log_j_pred

# ============================================================================
# BACKWARD COMPATIBILITY (optional - keeps original signature working)
# ============================================================================

def apply_depth_invariant_index_legacy(image_i, image_j, ki_kj, Ls):
    """
    Original function signature for backward compatibility.
    Assumes linear method with ki_kj as slope.
    """
    model = {'slope': ki_kj, 'intercept': 0}
    return apply_depth_invariant_index(image_i, image_j, model, Ls, method='linear')