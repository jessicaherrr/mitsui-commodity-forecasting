# preprocess_timeseries.py
import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, ifft, fftfreq
import warnings

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    pywt = None
    _HAS_PYWT = False

try:
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    _HAS_SK = True
except ImportError:
    KNNImputer = None
    StandardScaler = None
    MinMaxScaler = None
    RobustScaler = None
    _HAS_SK = False

def save_processed_data(df: pd.DataFrame, filename: str, output_dir: str = "processed_data") -> str:
    """
    Save processed DataFrame to CSV file
    
    Args:
        df: DataFrame to save
        filename: Output filename
        output_dir: Output directory (default: "processed_data")
        
    Returns:
        Full path to the saved file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Create full file path
    file_path = os.path.join(output_dir, filename)
    
    # Save DataFrame to CSV
    df.to_csv(file_path, index=False)
    
    print(f"✅ Processed data saved to: {file_path}")
    print(f"   Shape: {df.shape}")
    
    return file_path

# def load_csvs(base_path: str):
#     """Load the four CSV files expected: target_pairs.csv, train_labels.csv, train.csv, test.csv"""
#     files = {
#         'target_pairs': os.path.join(base_path, 'target_pairs.csv'),
#         'train_labels': os.path.join(base_path, 'train_labels.csv'),
#         'train': os.path.join(base_path, 'train.csv'),
#         'test': os.path.join(base_path, 'test.csv'),
#     }
#     dfs = {}
#     for k, path in files.items():
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"{path} not found.")
#         dfs[k] = pd.read_csv(path)
#     return dfs

def fill_missing(df: pd.DataFrame, strategy: str = 'interpolate', columns: list = None, knn_k: int = 5) -> pd.DataFrame:
    """Fill missing values with chosen strategy"""
    df = df.copy()
    cols = columns if columns is not None else df.columns.tolist()
    
    if strategy == 'ffill':
        df[cols] = df[cols].ffill().bfill()
    elif strategy == 'bfill':
        df[cols] = df[cols].bfill().ffill()
    elif strategy == 'interpolate':
        df[cols] = df[cols].interpolate(method='linear', limit_direction='both', axis=0)
        df[cols] = df[cols].ffill().bfill()
    elif strategy == 'median':
        for c in cols:
            df[c] = df[c].fillna(df[c].median())
    elif strategy == 'mean':
        for c in cols:
            df[c] = df[c].fillna(df[c].mean())
    elif strategy == 'knn':
        if not _HAS_SK:
            raise ImportError("sklearn not available for knn imputation.")
        imputer = KNNImputer(n_neighbors=knn_k)
        df[cols] = imputer.fit_transform(df[cols])
    elif strategy == 'drop':
        df = df.dropna(subset=cols)
    else:
        raise ValueError(f"Unknown missing value strategy '{strategy}'")
    return df

def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> tuple:
    """Returns boolean mask of outliers using IQR method, and lower/upper bounds."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    
    # Create separate boolean arrays and combine them safely
    below_lower = series < lower
    above_upper = series > upper
    
    # Use numpy logical_or for safe boolean operations
    mask = np.logical_or(below_lower.values, above_upper.values)
    
    return mask, lower, upper

def cap_outliers(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """Replace values outside [lower, upper] with the boundary values (winsorization)."""
    s = series.copy()
    s[s < lower] = lower
    s[s > upper] = upper
    return s

def wavelet_denoise(series: pd.Series, wavelet: str = 'db4', level: int = None, mode='soft') -> pd.Series:
    """Wavelet denoising using pywt."""
    if not _HAS_PYWT:
        raise ImportError("pywt package required for wavelet_denoise.")
    arr = series.astype(float).values
    if level is None:
        level = pywt.dwt_max_level(len(arr), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(arr, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs) > 1 else np.std(arr)
    uthresh = sigma * np.sqrt(2 * np.log(len(arr)))
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode=mode) for c in coeffs[1:]]
    rec = pywt.waverec(denoised_coeffs, wavelet)
    rec = rec[:len(arr)]
    return pd.Series(rec, index=series.index)

def gaussian_smooth(series: pd.Series, sigma: float = 2.0) -> pd.Series:
    """Gaussian smooth"""
    arr = series.astype(float).values
    rec = gaussian_filter1d(arr, sigma=sigma, mode='reflect')
    return pd.Series(rec, index=series.index)


def preprocess_pipeline(df: pd.DataFrame, columns_to_process: list = None, 
                       missing_strategy: str = 'interpolate', 
                       outlier_method: str = 'iqr', 
                       outlier_k: float = 1.5,
                       denoiser: str = 'gaussian',
                       denoiser_kwargs: dict = None) -> pd.DataFrame:
    """
    Complete preprocessing pipeline:
    - Fill missing values
    - Detect & cap outliers  
    - Apply denoising
    """
    denoiser_kwargs = denoiser_kwargs or {}
    
    # Determine columns to process
    cols = columns_to_process if columns_to_process is not None else [c for c in df.columns if c != 'date_id']
    
    # Create a complete copy of the input data including all columns
    df_out = df.copy()
    
    # 1. Handle missing values only for specified columns
    df_out[cols] = fill_missing(df_out, strategy=missing_strategy, columns=cols)[cols]
    
    # 2. Handle outliers only for specified columns
    for c in cols:
        if outlier_method == 'iqr':
            mask, lower, upper = detect_outliers_iqr(df_out[c], k=outlier_k)
            df_out[c] = cap_outliers(df_out[c], lower, upper)
        elif outlier_method == 'zscore':
            mean = df_out[c].mean()
            std = df_out[c].std()
            lower, upper = mean - outlier_k*std, mean + outlier_k*std
            df_out[c] = cap_outliers(df_out[c], lower, upper)
    
    # 3. Apply denoising - optimized to avoid fragmentation
    if denoiser and denoiser != 'none':
        print(f"Applying {denoiser} denoising to {len(cols)} columns...")
        
        successful_denoising = 0
        for c in cols:
            try:
                series = df_out[c].astype(float)
                
                if denoiser == 'wavelet':
                    denoised_series = wavelet_denoise(series, **denoiser_kwargs)
                elif denoiser == 'gaussian':
                    denoised_series = gaussian_smooth(series, **denoiser_kwargs)
                else:
                    denoised_series = series  # Fallback to original
                
                # Direct column replacement to avoid memory fragmentation
                df_out[c] = denoised_series
                successful_denoising += 1
                
            except Exception as e:
                warnings.warn(f"Denoiser {denoiser} failed for column {c}: {e}. Using original series.")
                # Keep original column unchanged
        
        print(f"Successfully denoised {successful_denoising}/{len(cols)} columns")
    
    return df_out


def safe_to_numeric(df: pd.DataFrame, errors: str = 'coerce') -> pd.DataFrame:
    """Convert all columns (except 'date_id') to numeric where possible."""
    df = df.copy()
    for col in df.columns:
        if col == 'date_id':
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors=errors)
    return df

def process_test_data(test_data: pd.DataFrame, 
                     train_columns: list = None,
                     missing_strategy: str = 'interpolate',
                     denoiser: str = 'gaussian',
                     denoiser_kwargs: dict = None) -> pd.DataFrame:
    """
    Process test data with special handling to ensure compatibility with training data
    
    Args:
        test_data: Raw test data DataFrame
        train_columns: Columns from training data to ensure feature alignment
        missing_strategy: Strategy for handling missing values
        denoiser: Denoising method
        denoiser_kwargs: Parameters for denoising method
        
    Returns:
        Processed test data DataFrame
    """
    print("Processing test data with enhanced compatibility...")
    
    # Create a copy of test data
    test_processed = test_data.copy()
    
    # Ensure numeric types
    test_processed = safe_to_numeric(test_processed)
    
    # If train_columns is provided, align features
    if train_columns is not None:
        test_processed = align_test_features(test_processed, train_columns)
    
    # Apply preprocessing (skip outlier detection for test data to avoid issues)
    test_processed = preprocess_pipeline(
        test_processed,
        columns_to_process=None,  # Process all numeric columns except 'date_id'
        missing_strategy=missing_strategy,
        outlier_method='none',    # Skip outlier detection for test data
        outlier_k=1.5,
        denoiser=denoiser,
        denoiser_kwargs=denoiser_kwargs
    )
    
    print(f"✅ Test data processing completed. Shape: {test_processed.shape}")
    return test_processed

def align_test_features(test_data: pd.DataFrame, train_columns: list) -> pd.DataFrame:
    """
    Align test data features with training data features
    
    Args:
        test_data: Test data DataFrame
        train_columns: Column names from training data
        
    Returns:
        Test data with aligned features
    """
    print("Aligning test features with training data...")
    
    # Get current test columns
    test_columns = test_data.columns.tolist()
    
    # Find missing columns in test data
    missing_in_test = set(train_columns) - set(test_columns)
    
    # Find extra columns in test data
    extra_in_test = set(test_columns) - set(train_columns)
    
    # Add missing columns with NaN values
    for col in missing_in_test:
        if col != 'date_id':  # Don't add date_id if it's not in test
            test_data[col] = np.nan
            print(f"   Added missing column: {col}")
    
    # Remove extra columns (keep date_id if present)
    columns_to_keep = [col for col in test_data.columns if col in train_columns or col == 'date_id']
    test_data = test_data[columns_to_keep]
    
    # Reorder columns to match training data (keep date_id first if present)
    if 'date_id' in test_data.columns and 'date_id' in train_columns:
        # Put date_id first, then follow training data order
        final_columns = ['date_id'] + [col for col in train_columns if col != 'date_id']
    else:
        final_columns = [col for col in train_columns if col in test_data.columns]
    
    test_data = test_data[final_columns]
    
    print(f"   Aligned features: {len(test_data.columns)} columns")
    if missing_in_test:
        print(f"   Added {len(missing_in_test)} missing columns")
    if extra_in_test:
        print(f"   Removed {len(extra_in_test)} extra columns")
    
    return test_data

def load_and_process_test_data(test_file_path: str, 
                              train_processed_path: str = None,
                              output_dir: str = "processed_data") -> pd.DataFrame:
    """
    Complete test data processing pipeline
    
    Args:
        test_file_path: Path to raw test data CSV
        train_processed_path: Path to processed training data (for feature alignment)
        output_dir: Output directory for processed test data
        
    Returns:
        Processed test data DataFrame
    """
    try:
        # Load test data
        print(f"Loading test data from: {test_file_path}")
        test_data = pd.read_csv(test_file_path)
        print(f"Raw test data shape: {test_data.shape}")
        
        # Load training data for feature alignment if provided
        train_columns = None
        if train_processed_path and os.path.exists(train_processed_path):
            train_processed = pd.read_csv(train_processed_path)
            train_columns = train_processed.columns.tolist()
            print(f"Using {len(train_columns)} features from training data for alignment")
        
        # Process test data
        test_processed = process_test_data(
            test_data=test_data,
            train_columns=train_columns,
            missing_strategy='interpolate',
            denoiser='gaussian',
            denoiser_kwargs={'sigma': 2.0}
        )
        
        # Save processed test data
        test_output_path = save_processed_data(test_processed, 'test_processed.csv', output_dir)
        
        print(f"✅ Test data processing completed successfully!")
        print(f"   Input shape: {test_data.shape}")
        print(f"   Output shape: {test_processed.shape}")
        print(f"   Saved to: {test_output_path}")
        
        return test_processed
        
    except Exception as e:
        print(f"❌ Error processing test data: {e}")
        raise

def validate_test_data_compatibility(test_processed: pd.DataFrame, 
                                   train_processed: pd.DataFrame) -> bool:
    """
    Validate that test data is compatible with training data
    
    Args:
        test_processed: Processed test data
        train_processed: Processed training data
        
    Returns:
        Boolean indicating compatibility
    """
    print("Validating test data compatibility...")
    
    # Check if all training features are present in test data
    train_features = [col for col in train_processed.columns if col != 'date_id']
    test_features = [col for col in test_processed.columns if col != 'date_id']
    
    missing_features = set(train_features) - set(test_features)
    extra_features = set(test_features) - set(train_features)
    
    if missing_features:
        print(f"❌ Missing features in test data: {len(missing_features)}")
        for feature in list(missing_features)[:5]:  # Show first 5
            print(f"   - {feature}")
        return False
    
    if extra_features:
        print(f"⚠️  Extra features in test data: {len(extra_features)}")
        for feature in list(extra_features)[:5]:  # Show first 5
            print(f"   - {feature}")
        # Extra features are usually OK, but we log them
    
    print(f"✅ Test data is compatible with training data")
    print(f"   Common features: {len(set(train_features) & set(test_features))}")
    return True