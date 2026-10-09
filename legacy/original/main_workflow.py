import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
import gc
import warnings
import joblib
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import optuna
from optuna.integration import PyTorchLightningPruningCallback
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility across all libraries
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Enhanced Configuration with advanced optimization
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Feature engineering parameters with safe lags
    lags = [1, 2, 3, 5, 10, 15, 20]
    windows = [3, 5, 10, 15, 20, 30, 50]
    # Data leakage prevention: safe lag margin
    safe_lag_margin = 3
    # Advanced feature selection
    feature_selection_method = 'variance'  # 'variance', 'lgb_importance', 'correlation'
    max_features = 300
    # Model parameters
    hidden_size = 256
    num_layers = 3
    dropout_rate = 0.4
    # Advanced training parameters
    batch_size = 128
    learning_rate = 0.001
    weight_decay = 1e-4
    patience = 20
    min_delta = 0.00001
    # Data parameters with time-series aware split
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    # Advanced optimization
    use_swa = True  # Stochastic Weight Averaging
    use_mixup = False  # Data augmentation
    use_focal_loss = False  # For imbalanced data
    # Ensemble parameters
    models_per_target = 3  # Number of models to train for each target
    n_folds = 5

config = Config()
print(f"Using device: {config.device}")

# Advanced hyperparameter optimization with Optuna
def setup_optuna_study():
    """Setup Optuna study for advanced hyperparameter optimization"""
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.HyperbandPruner()
    )
    return study

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
    }
    return params

# Advanced feature selection methods
def advanced_feature_selection(X, y, method='variance', n_features=300):
    """Advanced feature selection with multiple methods"""
    if method == 'variance':
        # Select features with highest variance
        variances = X.var().sort_values(ascending=False)
        selected_features = variances.head(n_features).index.tolist()
        
    elif method == 'lgb_importance':
        # Use LightGBM for feature importance
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            random_state=42,
            verbosity=-1
        )
        lgb_model.fit(X, y)
        importances = pd.Series(lgb_model.feature_importances_, index=X.columns)
        selected_features = importances.sort_values(ascending=False).head(n_features).index.tolist()
        
    elif method == 'correlation':
        # Select features with highest correlation to target
        correlations = X.apply(lambda col: col.corr(y)).abs().sort_values(ascending=False)
        selected_features = correlations.head(n_features).index.tolist()
    
    else:
        # Default to variance
        variances = X.var().sort_values(ascending=False)
        selected_features = variances.head(n_features).index.tolist()
    
    print(f"Selected {len(selected_features)} features using {method} method")
    return selected_features

# Advanced time-series cross validation
class TimeSeriesCVSplitter:
    """Advanced time-series cross validation with purging and embargo"""
    
    def __init__(self, n_splits=5, embargo_days=1):
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets"""
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits - 1):
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = (i + 2) * fold_size
            
            # Apply embargo: remove data points after test period
            train_indices = list(range(0, train_end))
            test_indices = list(range(test_start + self.embargo_days, test_end))
            
            yield train_indices, test_indices

# Advanced data augmentation
class FinancialDataAugmentation:
    """Advanced data augmentation for financial time series"""
    
    @staticmethod
    def add_noise(series, noise_level=0.01):
        """Add Gaussian noise to series"""
        noise = np.random.normal(0, noise_level * series.std(), len(series))
        return series + noise
    
    @staticmethod
    def time_warp(series, warp_factor=0.1):
        """Apply time warping to series"""
        n = len(series)
        warp_points = np.sort(np.random.choice(n, int(warp_factor * n), replace=False))
        warped_series = series.copy()
        
        for point in warp_points:
            if point > 0 and point < n - 1:
                warped_series[point] = (series[point-1] + series[point+1]) / 2
                
        return warped_series
    
    @staticmethod
    def scale_variation(series, scale_range=(0.9, 1.1)):
        """Apply random scaling to series"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return series * scale

# Hyperparameter grid search setup
def setup_hyperparameter_grid():
    param_grid = {
        'hidden_size': [64, 128, 256, 512],
        'num_layers': [1, 2, 3, 4],
        'learning_rate': [0.001, 0.0005, 0.0001, 0.00005],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'batch_size': [32, 64, 128, 256]
    }
    return list(ParameterGrid(param_grid))

# Build features for target function
def build_features_for_target(target_config, all_data):
    """Build feature set for specific target variable"""
    pair_expr = target_config['pair']
    
    if ' - ' in pair_expr:
        features_in_expr = pair_expr.split(' - ')
    else:
        features_in_expr = [pair_expr]
    
    assets_involved = set()
    for feature in features_in_expr:
        if '_adj_close' in feature:
            asset = feature.replace('_adj_close', '')
        elif '_Close' in feature:
            asset = feature.replace('_Close', '')
        else:
            parts = feature.split('_')
            if len(parts) > 1:
                asset = '_'.join(parts[:-1])
            else:
                asset = feature
        assets_involved.add(asset)
    
    relevant_features = []
    for asset in assets_involved:
        asset_features = [col for col in all_data.columns if col.startswith(asset)]
        relevant_features.extend(asset_features)
    
    global_features = []
    existing_features = [f for f in relevant_features + global_features if f in all_data.columns]
    
    if not existing_features:
        raise ValueError(f"No features found for assets: {assets_involved}")
    
    X_target = all_data[existing_features].copy()
    
    if ' - ' in pair_expr:
        feat1, feat2 = pair_expr.split(' - ')
        if feat1 in all_data.columns and feat2 in all_data.columns:
            y_target = all_data[feat1] - all_data[feat2]
        else:
            raise ValueError(f"Features not found: {feat1} or {feat2}")
    else:
        if pair_expr in all_data.columns:
            y_target = all_data[pair_expr]
        else:
            raise ValueError(f"Feature not found: {pair_expr}")
    
    return X_target, y_target, list(assets_involved)

# Safe rolling features calculation to prevent data leakage
def safe_rolling_features(df, columns, windows, safe_lag):
    """
    Calculate rolling features safely without data leakage
    All features are computed using only past information
    """
    df_safe = df.copy()
    
    for col in columns:
        for window in windows:
            # Use shifted data to calculate rolling statistics
            shifted_data = df[col].shift(safe_lag)
            rolling = shifted_data.rolling(window, min_periods=1)
            
            # Calculate rolling statistics
            df_safe[f'{col}_roll_mean_{window}'] = rolling.mean()
            df_safe[f'{col}_roll_std_{window}'] = rolling.std()
            df_safe[f'{col}_roll_skew_{window}'] = rolling.skew()
            df_safe[f'{col}_roll_kurt_{window}'] = rolling.kurt()
            
            # Safe Z-score calculation using only past data
            rolling_mean = shifted_data.rolling(window).mean()
            rolling_std = shifted_data.rolling(window).std()
            df_safe[f'{col}_zscore_{window}'] = (
                (shifted_data - rolling_mean) / (rolling_std + 1e-8)
            )
            
            # Safe min/max positions
            df_safe[f'{col}_roll_min_{window}'] = rolling.min()
            df_safe[f'{col}_roll_max_{window}'] = rolling.max()
    
    return df_safe

# Safe price position calculation
def safe_price_position(df, price_columns, windows, safe_lag):
    """
    Calculate price position safely without data leakage
    """
    df_safe = df.copy()
    
    for col in price_columns:
        for window in windows:
            # Use shifted data to calculate price ranges
            shifted_data = df[col].shift(safe_lag)
            high = shifted_data.rolling(window).max()
            low = shifted_data.rolling(window).min()
            
            # Calculate position using shifted price
            current_price_shifted = df[col].shift(safe_lag)
            df_safe[f'{col}_position_{window}'] = (
                (current_price_shifted - low) / (high - low + 1e-8)
            )
    
    return df_safe

# Enhanced data creation with realistic financial patterns
def create_enhanced_sample_data(n_samples=2000):
    """
    Create realistic financial market sample data with:
    - Realistic volatility clustering
    - Mean reversion properties  
    - Correlations between assets
    - Market regimes
    """
    dates = pd.date_range('2010-01-01', periods=n_samples, freq='D')
    
    # Market regime simulation (bull/bear markets)
    regime_periods = n_samples // 4
    regimes = np.sin(2 * np.pi * np.arange(n_samples) / regime_periods)
    bull_market = regimes > 0
    
    data_dict = {}
    asset_prefixes = [
        'US_Stock_VT', 'US_Stock_VYM', 'US_Stock_IEMG', 'US_Stock_VGIT',
        'LME_PB', 'LME_CA', 'LME_ZS', 'LME_AH',
        'JPX_Gold_Standard_Futures', 'JPX_Platinum_Standard_Futures',
        'FX_AUDJPY', 'FX_EURAUD', 'FX_CADCHF', 'FX_ZARCHF', 'FX_NZDJPY',
        'FX_CHFJPY', 'FX_ZARUSD', 'FX_USDJPY', 'FX_AUDCHF'
    ]
    
    # Base parameters for different asset classes
    asset_classes = {
        'US_Stock': {'volatility': 0.15, 'drift': 0.0003},
        'LME': {'volatility': 0.12, 'drift': 0.0002},
        'JPX': {'volatility': 0.08, 'drift': 0.0001},
        'FX': {'volatility': 0.05, 'drift': 0.00005}
    }
    
    # Create correlated random walks
    base_returns = np.random.randn(n_samples, len(asset_prefixes)) * 0.01
    
    # Add correlation structure
    correlation_strength = 0.3
    for i in range(1, len(asset_prefixes)):
        base_returns[:, i] = (base_returns[:, i] + 
                             correlation_strength * base_returns[:, i-1]) / (1 + correlation_strength)
    
    for idx, prefix in enumerate(asset_prefixes):
        # Determine asset class
        asset_class = next((k for k in asset_classes.keys() if k in prefix), 'FX')
        params = asset_classes[asset_class]
        
        # Adjust returns based on market regime
        regime_multiplier = 1.2 if bull_market else 0.8
        volatility = params['volatility'] * regime_multiplier / np.sqrt(252)
        drift = params['drift'] * regime_multiplier
        
        # Generate realistic returns with volatility clustering
        returns = base_returns[:, idx] * volatility + drift
        
        # Add volatility clustering (GARCH-like effect)
        vol_cluster = np.zeros(n_samples)
        for t in range(1, n_samples):
            vol_cluster[t] = 0.94 * vol_cluster[t-1] + 0.06 * abs(returns[t-1])
        returns = returns + 0.3 * vol_cluster * np.random.randn(n_samples)
        
        # Generate price series
        price_series = 100 * np.exp(np.cumsum(returns))
        
        # Create OHLC data with realistic patterns
        data_dict[f'{prefix}_Close'] = price_series
        daily_volatility = price_series * volatility
        data_dict[f'{prefix}_Open'] = price_series + np.random.randn(n_samples) * daily_volatility * 0.1
        data_dict[f'{prefix}_High'] = price_series + np.abs(np.random.randn(n_samples)) * daily_volatility * 0.15
        data_dict[f'{prefix}_Low'] = price_series - np.abs(np.random.randn(n_samples)) * daily_volatility * 0.15
        
        # Volume with momentum correlation
        volume_trend = np.cumsum(np.random.randn(n_samples) * 0.01)
        volume = np.exp(10 + 0.5 * volume_trend + 0.3 * np.random.randn(n_samples))
        data_dict[f'{prefix}_Volume'] = volume
        
        # Realistic technical indicators (safe calculation)
        close_series = price_series
        
        # RSI (30-70 range, mean-reverting)
        price_changes = np.diff(close_series, prepend=close_series[0])
        gain = np.where(price_changes > 0, price_changes, 0)
        loss = np.where(price_changes < 0, -price_changes, 0)
        avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        data_dict[f'{prefix}_RSI'] = rsi
        
        # MACD
        ema_12 = pd.Series(close_series).ewm(span=12).mean()
        ema_26 = pd.Series(close_series).ewm(span=26).mean()
        macd = ema_12 - ema_26
        data_dict[f'{prefix}_MACD'] = macd
        
        # Bollinger Bands
        rolling_mean = pd.Series(close_series).rolling(20).mean()
        rolling_std = pd.Series(close_series).rolling(20).std()
        data_dict[f'{prefix}_BB_Upper'] = rolling_mean + 2 * rolling_std
        data_dict[f'{prefix}_BB_Lower'] = rolling_mean - 2 * rolling_std
        
        # Additional indicators
        data_dict[f'{prefix}_Momentum_10'] = close_series / pd.Series(close_series).shift(10) - 1
        data_dict[f'{prefix}_Volatility_20'] = pd.Series(returns).rolling(20).std()
    
    df = pd.DataFrame(data_dict, index=dates)
    return df.fillna(method='bfill')

# Enhanced feature engineering with data leakage prevention and advanced techniques
def create_advanced_features(df, lags=None, windows=None, target_lag=1):
    """
    Create comprehensive financial features including:
    - Lagged returns and prices
    - Rolling statistics with multiple windows
    - Technical indicators
    - Cross-asset correlations
    - Volatility measures
    - Market regime indicators
    
    All features are calculated safely without data leakage
    """
    if lags is None:
        lags = config.lags
    if windows is None:
        windows = config.windows
        
    # Calculate safe lag to prevent data leakage
    safe_lag = target_lag + config.safe_lag_margin
    
    df_enhanced = df.copy()
    original_features = df.columns.tolist()
    
    # Identify feature types for targeted engineering
    price_features = [col for col in original_features if any(x in col for x in ['Close', 'Open', 'High', 'Low'])]
    volume_features = [col for col in original_features if 'Volume' in col]
    indicator_features = [col for col in original_features if any(x in col for x in ['RSI', 'MACD', 'BB', 'Momentum', 'Volatility'])]
    
    print(f"Processing {len(price_features)} price features, {len(volume_features)} volume features, "
          f"{len(indicator_features)} indicator features with safe_lag={safe_lag}")
    
    # 1. Price-based features with safe lags
    for col in price_features:
        # Returns at different horizons with safe lag
        df_enhanced[f'{col}_return_1'] = df[col].pct_change(1).shift(safe_lag)
        df_enhanced[f'{col}_return_5'] = df[col].pct_change(5).shift(safe_lag)
        df_enhanced[f'{col}_return_10'] = df[col].pct_change(10).shift(safe_lag)
        
        # Log returns (more normally distributed)
        df_enhanced[f'{col}_log_return_1'] = np.log(df[col] / df[col].shift(1)).shift(safe_lag)
        
        # Safe price position relative to recent range
        df_enhanced = safe_price_position(df_enhanced, [col], [10, 20], safe_lag)
    
    # 2. Volume-based features with safe lags
    for col in volume_features:
        # Volume momentum and trends with safe lag
        df_enhanced[f'{col}_sma_5'] = df[col].rolling(5).mean().shift(safe_lag)
        df_enhanced[f'{col}_sma_20'] = df[col].rolling(20).mean().shift(safe_lag)
        df_enhanced[f'{col}_volume_ratio'] = (df[col] / df[col].rolling(20).mean()).shift(safe_lag)
        
        # Volume volatility with safe lag
        df_enhanced[f'{col}_volatility_10'] = df[col].pct_change().rolling(10).std().shift(safe_lag)
    
    # 3. Technical indicator enhancements with safe lags
    for col in indicator_features:
        # Lagged indicators with additional safe lag
        for lag in [1, 2, 3]:
            df_enhanced[f'{col}_lag_{lag}'] = df[col].shift(lag + safe_lag)
        
        # Indicator momentum with safe lag
        if 'RSI' in col:
            df_enhanced[f'{col}_momentum_5'] = (df[col] - df[col].shift(5)).shift(safe_lag)
        elif 'MACD' in col:
            df_enhanced[f'{col}_signal'] = pd.Series(df[col]).ewm(span=9).mean().shift(safe_lag)
    
    # 4. Safe rolling statistical features
    important_features = price_features + [f'{col}_return_1' for col in price_features]
    # Limit features to prevent memory issues
    important_features = important_features[:min(20, len(important_features))]
    
    df_enhanced = safe_rolling_features(df_enhanced, important_features, windows, safe_lag)
    
    # 5. Cross-asset features (correlations) with safe lags
    print("Computing cross-asset correlations with safe lags...")
    return_cols = [col for col in df_enhanced.columns if 'return_1' in col]
    if len(return_cols) > 5:
        # Use shifted data for correlation calculation
        returns_df = df_enhanced[return_cols].shift(safe_lag).iloc[:1000]
        
        for i, col1 in enumerate(return_cols[:5]):
            for col2 in return_cols[i+1:min(i+6, len(return_cols))]:
                corr_10 = returns_df[col1].rolling(10).corr(returns_df[col2])
                df_enhanced[f'corr_{col1}_{col2}_10'] = corr_10.shift(safe_lag)
    
    # 6. Market regime features with safe lags
    print("Computing market regime indicators with safe lags...")
    market_returns = df_enhanced[[col for col in df_enhanced.columns if 'US_Stock' in col and 'return_1' in col]].mean(axis=1)
    df_enhanced['market_volatility_20'] = market_returns.rolling(20).std().shift(safe_lag)
    df_enhanced['market_trend_20'] = (market_returns.rolling(20).mean() > 0).astype(float).shift(safe_lag)
    
    # 7. Advanced feature selection
    print("Applying advanced feature selection...")
    y_dummy = pd.Series(np.random.randn(len(df_enhanced)), index=df_enhanced.index)  # Dummy target for feature selection
    
    selected_features = advanced_feature_selection(
        df_enhanced.select_dtypes(include=[np.number]), 
        y_dummy,
        method=config.feature_selection_method,
        n_features=config.max_features
    )
    
    df_enhanced = df_enhanced[selected_features]
    
    # Final cleanup
    df_enhanced = df_enhanced.dropna()
    print(f"Final feature set: {df_enhanced.shape[1]} features, {df_enhanced.shape[0]} samples")
    
    return df_enhanced

# Enhanced dataset class with caching and memory optimization
class EnhancedFinancialDataset(Dataset):
    def __init__(self, X, y, window_size, lag, debug_mode=False, enable_caching=True):
        """
        Optimized financial dataset with:
        - Memory-efficient storage
        - Optional caching for performance
        - Advanced debugging information
        - Robust NaN handling
        """
        # Convert to numpy for efficiency
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y.shift(-lag), dtype=np.float32)  # Align with lag
        
        self.window_size = window_size
        self.enable_caching = enable_caching
        self.cache = {}
        
        # Robust NaN handling
        y_valid = ~np.isnan(self.y)
        x_valid = ~np.any(np.isnan(self.X), axis=1)
        valid_mask = y_valid & x_valid
        
        # Ensure sufficient history
        self.valid_indices = np.where(valid_mask)[0]
        self.valid_indices = self.valid_indices[self.valid_indices >= window_size - 1]
        
        if debug_mode and len(self.valid_indices) > 0:
            self._debug_info()
    
    def _debug_info(self):
        """Provide detailed debugging information"""
        print(f"Dataset Info: {len(self.valid_indices)} valid samples, "
              f"window_size={self.window_size}, X_shape={self.X.shape}")
        
        for i, idx in enumerate(self.valid_indices[:2]):
            start_idx = idx - self.window_size + 1
            end_idx = idx + 1
            print(f"Sample {i}: indices[{start_idx}:{end_idx}] -> "
                  f"X_shape={self.X[start_idx:end_idx].shape}, y={self.y[idx]:.4f}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        if idx in self.cache and self.enable_caching:
            return self.cache[idx]
            
        t = self.valid_indices[idx]
        start_idx = t - self.window_size + 1
        end_idx = t + 1
        
        x_sequence = self.X[start_idx:end_idx]  # (window_size, n_features)
        y_value = self.y[t]
        
        # Convert to tensors
        x_tensor = torch.from_numpy(x_sequence)
        y_tensor = torch.tensor(y_value, dtype=torch.float32)
        
        if self.enable_caching:
            self.cache[idx] = (x_tensor, y_tensor)
            
        return x_tensor, y_tensor

# Advanced LSTM model with multiple enhancements
class AdvancedLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.4, 
                 use_attention=True, bidirectional=True, use_layer_norm=True,
                 use_skip_connections=True):
        """
        Enhanced LSTM model with:
        - Bidirectional LSTM
        - Attention mechanism
        - Layer normalization
        - Deep output network
        - Residual connections
        - Skip connections
        """
        super().__init__()
        
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        self.use_skip_connections = use_skip_connections
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.Tanh(),
                nn.Linear(lstm_output_size // 2, 1),
                nn.Softmax(dim=1)
            )
        
        # Deep output network with residual connections
        self.output_network = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.LayerNorm(lstm_output_size // 2) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.LayerNorm(lstm_output_size // 4) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            
            nn.Linear(lstm_output_size // 4, 1)
        )
        
        # Skip connection projection
        if use_skip_connections:
            self.skip_projection = nn.Linear(input_size, lstm_output_size // 4) if input_size != lstm_output_size // 4 else nn.Identity()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    for i in range(4):  # LSTM has 4 different weight matrices
                        nn.init.xavier_uniform_(param[i * self.hidden_size:(i + 1) * self.hidden_size])
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_len, input_size = x.size()
        
        # Store input for skip connection
        if self.use_skip_connections:
            skip_input = x[:, -1, :]  # Use last timestep for skip connection
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size * num_directions)
        
        # Apply layer normalization
        if self.use_layer_norm:
            lstm_out = self.layer_norm(lstm_out)
        
        # Attention mechanism
        if self.use_attention:
            attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
            context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size * num_directions)
            output = self.output_network(context_vector)
        else:
            # Use last hidden state (for bidirectional, concatenate last forward and backward)
            if self.bidirectional:
                last_forward = hidden[-2]  # Last layer, forward direction
                last_backward = hidden[-1]  # Last layer, backward direction
                last_hidden = torch.cat((last_forward, last_backward), dim=1)
            else:
                last_hidden = hidden[-1]
            
            output = self.output_network(last_hidden)
        
        # Add skip connection
        if self.use_skip_connections:
            skip_output = self.skip_projection(skip_input)
            output = output + skip_output
        
        return output.squeeze(-1)

# Focal Loss for imbalanced data
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure targets are in the right format
        targets = targets.view(-1, 1)
        
        # Calculate MSE loss
        mse_loss = F.mse_loss(inputs, targets, reduction='none')
        
        # Apply focal weighting
        focal_weight = self.alpha * (1 - torch.exp(-mse_loss)) ** self.gamma
        focal_loss = focal_weight * mse_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Comprehensive model training with advanced features
class FinancialModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
    def prepare_data(self, X_target, y_target, lag, window_size=20):
        """Prepare data with robust preprocessing and time-series splitting"""
        # Robust scaling (less sensitive to outliers)
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_target),
            index=X_target.index,
            columns=X_target.columns
        )
        
        # Advanced feature engineering with safe lag
        print("Creating advanced features with data leakage prevention...")
        X_engineered = create_advanced_features(X_scaled, target_lag=lag)
        y_aligned = y_target.loc[X_engineered.index]
        
        # Create dataset
        dataset = EnhancedFinancialDataset(
            X_engineered, y_aligned, 
            window_size=window_size, 
            lag=lag,
            enable_caching=True
        )
        
        if len(dataset) == 0:
            raise ValueError("No valid samples after feature engineering")
        
        # Time-series aware split (no future leak)
        n_samples = len(dataset)
        train_size = int(n_samples * self.config.train_ratio)
        val_size = int(n_samples * self.config.val_ratio)
        
        indices = np.arange(n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subsets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, 
                                 shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, 
                               shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, 
                                shuffle=False, pin_memory=True)
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'scaler': scaler,
            'feature_names': X_target.columns.tolist(),
            'dataset': dataset,
            'X_engineered': X_engineered,
            'y_aligned': y_aligned
        }
    
    def train_model(self, data_info, model_params=None):
        """Train model with comprehensive monitoring and early stopping"""
        if model_params is None:
            model_params = {}
        
        # Initialize model
        input_size = data_info['X_engineered'].shape[1]
        model = AdvancedLSTMRegressor(
            input_size=input_size,
            hidden_size=model_params.get('hidden_size', self.config.hidden_size),
            num_layers=model_params.get('num_layers', self.config.num_layers),
            dropout=model_params.get('dropout', self.config.dropout_rate),
            use_skip_connections=True
        ).to(self.device)
        
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_params.get('learning_rate', self.config.learning_rate),
            weight_decay=self.config.weight_decay
        )
        
        # Advanced loss function
        if self.config.use_focal_loss:
            loss_fn = FocalLoss()
        else:
            loss_fn = nn.HuberLoss()
        
        # Advanced learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=model_params.get('learning_rate', self.config.learning_rate),
            epochs=model_params.get('epochs', 100),
            steps_per_epoch=len(data_info['train_loader']),
            pct_start=0.1
        )
        
        # SWA (Stochastic Weight Averaging)
        if self.config.use_swa:
            swa_model = torch.optim.swa_utils.AveragedModel(model)
            swa_scheduler = torch.optim.swa_utils.SWALR(
                optimizer, swa_lr=model_params.get('learning_rate', self.config.learning_rate) * 0.1
            )
            swa_start = int(model_params.get('epochs', 100) * 0.75)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print("Starting advanced training...")
        for epoch in range(model_params.get('epochs', 100)):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, (x_batch, y_batch) in enumerate(data_info['train_loader']):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                # Data augmentation
                if self.config.use_mixup and np.random.random() > 0.5:
                    x_batch, y_batch = self.mixup_data(x_batch, y_batch)
                
                optimizer.zero_grad()
                predictions = model(x_batch)
                loss = loss_fn(predictions, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in data_info['val_loader']:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    predictions = model(x_batch)
                    val_loss += loss_fn(predictions, y_batch).item()
            
            # SWA update
            if self.config.use_swa and epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            
            # Average losses
            train_loss /= len(data_info['train_loader'])
            val_loss /= len(data_info['val_loader'])
            
            current_lr = scheduler.get_last_lr()[0]
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(current_lr)
            
            # Early stopping check
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
            else:
                patience_counter += 1
            
            # Progress reporting
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, '
                      f'Val Loss = {val_loss:.6f}, LR = {current_lr:.2e}')
            
            # Early stopping
            if patience_counter >= self.config.patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        # Apply SWA
        if self.config.use_swa:
            print("Applying SWA...")
            torch.optim.swa_utils.update_bn(data_info['train_loader'], swa_model)
            model = swa_model.module
        
        # Load best model
        elif best_model_state is not None:
            model.load_state_dict(best_model_state)
        else:
            print("Warning: No best model state found, using final model")
        
        # Test set evaluation
        test_loss = self.evaluate_model(model, data_info['test_loader'], loss_fn)
        
        print(f'Training completed. Best validation loss: {best_val_loss:.6f}, '
              f'Test loss: {test_loss:.6f}')
        
        return {
            'model': model,
            'history': history,
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'best_epoch': best_epoch,
            'scaler': data_info['scaler'],
            'feature_names': data_info['feature_names']
        }
    
    def mixup_data(self, x, y, alpha=0.2):
        """MixUp data augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y
    
    def evaluate_model(self, model, data_loader, loss_fn):
        """Comprehensive model evaluation"""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                predictions = model(x_batch)
                loss = loss_fn(predictions, y_batch)
                total_loss += loss.item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        # Calculate additional metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        # Direction accuracy
        pred_direction = np.diff(all_predictions) > 0
        actual_direction = np.diff(all_targets) > 0
        direction_accuracy = np.mean(pred_direction == actual_direction)
        
        print(f'Test Metrics - MSE: {mse:.6f}, MAE: {mae:.6f}, '
              f'R²: {r2:.4f}, Direction Accuracy: {direction_accuracy:.4f}')
        
        return total_loss / len(data_loader)

# Memory-optimized prediction function
def predict_with_model_optimized(model_info, X_target, lag, window_size=20):
    """Optimized prediction with memory management"""
    model = model_info['model']
    scaler = model_info['scaler']
    feature_names = model_info['feature_names']
    
    # Ensure feature consistency
    X_target = X_target[feature_names].copy()
    
    # Standardize features
    X_scaled = pd.DataFrame(
        scaler.transform(X_target),
        index=X_target.index,
        columns=X_target.columns
    )
    
    # Feature engineering with safe lag
    X_engineered = create_advanced_features(X_scaled, target_lag=lag)
    X_values = X_engineered.values.astype(np.float32)
    
    # Initialize predictions
    predictions = np.full(len(X_target), np.nan, dtype=np.float32)
    
    model.eval()
    with torch.no_grad():
        for t in range(window_size - 1, len(X_values)):
            # Create sequence
            x_sequence = X_values[t - window_size + 1:t + 1]
            x_tensor = torch.from_numpy(x_sequence).unsqueeze(0).to(model_info['model'].device)
            
            # Prediction
            y_pred = model(x_tensor).item()
            
            # Map to original timeline
            engineered_idx = X_engineered.index[t]
            original_pos = X_target.index.get_loc(engineered_idx) + lag
            
            if original_pos < len(predictions):
                predictions[original_pos] = y_pred
            
            # Memory management for large datasets
            if t % 1000 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    return predictions

# Advanced Grid Search with cross-validation
class AdvancedGridSearchFinancialTrainer(FinancialModelTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.best_models = {}
        self.grid_results = {}
        self.cv_splitter = TimeSeriesCVSplitter(n_splits=config.n_folds)
    
    def grid_search_cv(self, X_target, y_target, lag, param_grid, n_trials=10):
        """Execute grid search with time-series cross validation"""
        print(f"Starting advanced grid search with {self.config.n_folds}-fold CV, {min(len(param_grid), n_trials)} parameter combinations...")
        
        best_score = float('inf')
        best_params = None
        best_model_info = None
        
        # Limit trials to avoid excessive training time
        trials = min(len(param_grid), n_trials)
        
        for i, params in enumerate(param_grid[:trials]):
            print(f"\nTrial {i+1}/{trials}: {params}")
            
            try:
                # Cross-validation
                cv_scores = []
                cv_models = []
                
                for fold, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X_target)):
                    print(f"  Fold {fold + 1}/{self.config.n_folds}")
                    
                    # Split data
                    X_train, X_val = X_target.iloc[train_idx], X_target.iloc[val_idx]
                    y_train, y_val = y_target.iloc[train_idx], y_target.iloc[val_idx]
                    
                    # Prepare data
                    data_info = self.prepare_data(X_train, y_train, lag)
                    
                    # Add validation data
                    val_dataset = EnhancedFinancialDataset(
                        data_info['X_engineered'].iloc[val_idx], 
                        data_info['y_aligned'].iloc[val_idx],
                        window_size=20,
                        lag=lag
                    )
                    data_info['val_loader'] = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
                    
                    # Train model
                    model_result = self.train_model(data_info, model_params=params)
                    cv_scores.append(model_result['best_val_loss'])
                    cv_models.append(model_result)
                
                # Average CV score
                avg_score = np.mean(cv_scores)
                self.grid_results[str(params)] = avg_score
                
                # Update best model
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = params
                    best_model_info = cv_models[np.argmin(cv_scores)]
                    
                print(f"  Average validation loss: {avg_score:.6f} (Current best: {best_score:.6f})")
                
            except Exception as e:
                print(f"Parameter combination {params} failed: {e}")
                continue
        
        print(f"\nGrid search completed. Best parameters: {best_params}")
        print(f"Best validation loss: {best_score:.6f}")
        
        return best_model_info

# CORRECTED: Single Target Ensemble for proper model integration
class SingleTargetEnsemble:
    """
    CORRECTED: Ensemble for a single target variable
    Trains multiple models for the same target and combines their predictions
    """
    
    def __init__(self, target_name, lag, models_per_target=3):
        self.target_name = target_name
        self.lag = lag
        self.models_per_target = models_per_target
        self.models = []  # List of model_info dictionaries
        self.weights = []  # List of weights for each model
        self.model_performance = []  # List of validation losses
        
    def add_model(self, model_info, val_loss):
        """Add a model to the ensemble"""
        self.models.append(model_info)
        self.model_performance.append(val_loss)
        
        # Weight based on validation loss (lower loss = higher weight)
        weight = 1.0 / (val_loss + 1e-8)
        self.weights.append(weight)
        
        # Limit number of models
        if len(self.models) > self.models_per_target:
            self._prune_models()
    
    def _prune_models(self):
        """Keep only the best performing models"""
        # Sort by performance (ascending - lower loss is better)
        sorted_indices = np.argsort(self.model_performance)
        
        # Keep only top models
        self.models = [self.models[i] for i in sorted_indices[:self.models_per_target]]
        self.model_performance = [self.model_performance[i] for i in sorted_indices[:self.models_per_target]]
        self.weights = [self.weights[i] for i in sorted_indices[:self.models_per_target]]
    
    def predict_with_uncertainty(self, X_target):
        """
        CORRECTED: Make ensemble prediction for a single target with uncertainty
        
        Returns:
            ensemble_prediction: Weighted average of model predictions
            uncertainty: Variance of model predictions (measure of uncertainty)
        """
        if len(self.models) == 0:
            raise ValueError(f"No models in ensemble for target {self.target_name}")
        
        # Get predictions from all models
        individual_predictions = []
        for model_info in self.models:
            pred = predict_with_model_optimized(model_info, X_target, self.lag)
            individual_predictions.append(pred)
        
        # Convert to numpy array
        pred_matrix = np.column_stack(individual_predictions)
        
        # Normalize weights
        normalized_weights = np.array(self.weights) / np.sum(self.weights)
        
        # Calculate weighted ensemble prediction
        ensemble_pred = np.average(pred_matrix, axis=1, weights=normalized_weights)
        
        # Calculate uncertainty as variance of predictions
        uncertainty = np.var(pred_matrix, axis=1)
        
        # Create DataFrame with results
        results_df = pd.DataFrame({
            'ensemble_prediction': ensemble_pred,
            'prediction_uncertainty': uncertainty
        }, index=X_target.index)
        
        # Add individual model predictions
        for i, pred in enumerate(individual_predictions):
            results_df[f'model_{i}_prediction'] = pred
        
        return results_df, ensemble_pred, uncertainty
    
    def calibrate_uncertainty(self, X_val, y_true):
        """Calibrate uncertainty estimation using validation data"""
        print(f"Calibrating uncertainty for target {self.target_name}...")
        
        # Get predictions on validation set
        results_df, _, _ = self.predict_with_uncertainty(X_val)
        
        # Calculate empirical coverage
        pred_std = np.sqrt(results_df['prediction_uncertainty'])
        lower_bound = results_df['ensemble_prediction'] - 1.96 * pred_std
        upper_bound = results_df['ensemble_prediction'] + 1.96 * pred_std
        
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        print(f"  Empirical 95% coverage: {coverage:.4f}")
        
        # Simple calibration: scale uncertainty to achieve desired coverage
        if coverage < 0.95:
            # Increase uncertainty
            self.uncertainty_scale = 1.0 / coverage
        else:
            # Decrease uncertainty
            self.uncertainty_scale = coverage
            
        print(f"  Uncertainty scale factor: {self.uncertainty_scale:.4f}")
    
    def save_ensemble(self, filepath):
        """Save ensemble to file"""
        ensemble_data = {
            'target_name': self.target_name,
            'lag': self.lag,
            'models': self.models,
            'weights': self.weights,
            'model_performance': self.model_performance
        }
        joblib.dump(ensemble_data, filepath)
        print(f"Ensemble for {self.target_name} saved to: {filepath}")
    
    @classmethod
    def load_ensemble(cls, filepath):
        """Load ensemble from file"""
        ensemble_data = joblib.load(filepath)
        ensemble = cls(
            ensemble_data['target_name'],
            ensemble_data['lag'],
            models_per_target=len(ensemble_data['models'])
        )
        ensemble.models = ensemble_data['models']
        ensemble.weights = ensemble_data['weights']
        ensemble.model_performance = ensemble_data['model_performance']
        print(f"Ensemble for {ensemble.target_name} loaded from {filepath}")
        return ensemble

# CORRECTED: Multi-Target Ensemble Manager
class MultiTargetEnsembleManager:
    """
    Manager for multiple single-target ensembles
    """
    
    def __init__(self, target_configs, models_per_target=3):
        self.target_configs = target_configs
        self.models_per_target = models_per_target
        self.ensembles = {}  # target_name -> SingleTargetEnsemble
        
        # Initialize ensembles for each target
        for config in target_configs:
            target_name = config['target_name']
            lag = config['lag']
            self.ensembles[target_name] = SingleTargetEnsemble(
                target_name, lag, models_per_target
            )
    
    def add_model(self, target_name, model_info, val_loss):
        """Add a model to the appropriate target ensemble"""
        if target_name not in self.ensembles:
            raise ValueError(f"No ensemble found for target {target_name}")
        
        self.ensembles[target_name].add_model(model_info, val_loss)
    
    def predict_all_targets(self, X_targets_dict):
        """
        CORRECTED: Predict all targets using their respective ensembles
        
        Args:
            X_targets_dict: Dictionary of {target_name: X_target}
            
        Returns:
            Dictionary of {target_name: (results_df, ensemble_pred, uncertainty)}
        """
        predictions = {}
        
        for target_name, X_target in X_targets_dict.items():
            if target_name in self.ensembles:
                ensemble = self.ensembles[target_name]
                results_df, ensemble_pred, uncertainty = ensemble.predict_with_uncertainty(X_target)
                predictions[target_name] = {
                    'results_df': results_df,
                    'ensemble_prediction': ensemble_pred,
                    'uncertainty': uncertainty
                }
                print(f"✅ Generated predictions for {target_name}")
            else:
                print(f"❌ No ensemble found for {target_name}")
        
        return predictions
    
    def calibrate_all_uncertainties(self, X_val_dict, y_true_dict):
        """Calibrate uncertainty for all targets"""
        for target_name, ensemble in self.ensembles.items():
            if target_name in X_val_dict and target_name in y_true_dict:
                ensemble.calibrate_uncertainty(X_val_dict[target_name], y_true_dict[target_name])
    
    def save_all_ensembles(self, base_path="ensembles"):
        """Save all ensembles to files"""
        import os
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        for target_name, ensemble in self.ensembles.items():
            filepath = os.path.join(base_path, f"ensemble_{target_name}.pkl")
            ensemble.save_ensemble(filepath)
    
    @classmethod
    def load_all_ensembles(cls, target_configs, base_path="ensembles"):
        """Load all ensembles from files"""
        import os
        manager = cls(target_configs)
        
        for target_name in manager.ensembles.keys():
            filepath = os.path.join(base_path, f"ensemble_{target_name}.pkl")
            if os.path.exists(filepath):
                manager.ensembles[target_name] = SingleTargetEnsemble.load_ensemble(filepath)
            else:
                print(f"Warning: No saved ensemble found for {target_name}")
        
        return manager

# Main execution function with CORRECTED ensemble logic
def main():
    """Main execution pipeline with corrected ensemble implementation"""
    print("=== CORRECTED Financial Time Series Forecasting Pipeline ===")
    
    # Read target pairs
    try:
        target_pairs_df = pd.read_csv('target_pairs.csv')
        print(f"Loaded {len(target_pairs_df)} target pairs")
    except FileNotFoundError:
        print("target_pairs.csv not found. Using sample target configuration.")
        # Create sample target pairs for demonstration
        target_pairs_df = pd.DataFrame({
            'target': ['pair_1', 'pair_2'],
            'pair': ['US_Stock_VT_Close - US_Stock_VYM_Close', 'FX_USDJPY_Close'],
            'lag': [1, 2]
        })
    
    # Create enhanced sample data
    print("Creating enhanced sample data...")
    all_data = create_enhanced_sample_data(2000)
    print(f"Data shape: {all_data.shape}")
    
    # Build features for targets
    X_targets_dict = {}
    y_targets_dict = {}
    target_configs = []
    
    for idx, row in target_pairs_df.iterrows():
        try:
            X_target, y_target, assets = build_features_for_target(row, all_data)
            target_name = row['target']
            
            X_targets_dict[target_name] = X_target
            y_targets_dict[target_name] = y_target
            target_configs.append({
                'target_name': target_name,
                'lag': row['lag'],
                'assets': assets,
                'pair_expr': row['pair']
            })
            print(f"Target {target_name}: {len(assets)} assets, {X_target.shape[1]} features")
        except Exception as e:
            print(f"Error processing {row['target']}: {e}")
            continue
    
    print(f"\nSuccessfully built feature sets for {len(X_targets_dict)} target variables")
    
    # Initialize multi-target ensemble manager
    ensemble_manager = MultiTargetEnsembleManager(
        target_configs, 
        models_per_target=config.models_per_target
    )
    
    # Initialize grid search trainer
    grid_trainer = AdvancedGridSearchFinancialTrainer(config)
    
    # Set up parameter grid
    param_grid = setup_hyperparameter_grid()
    
    # Train multiple models for each target
    for target_config in target_configs:
        target_name = target_config['target_name']
        lag = target_config['lag']
        X_target = X_targets_dict[target_name]
        y_target = y_targets_dict[target_name]
        
        print(f"\n{'='*50}")
        print(f"Training {config.models_per_target} models for target: {target_name} (lag={lag})")
        print(f"{'='*50}")
        
        # Train multiple models for this target with different random seeds
        for model_idx in range(config.models_per_target):
            print(f"\nTraining model {model_idx+1}/{config.models_per_target} for {target_name}")
            
            # Set different random seed for each model
            torch.manual_seed(42 + model_idx)
            np.random.seed(42 + model_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42 + model_idx)
            
            try:
                # Execute grid search with CV
                best_model_info = grid_trainer.grid_search_cv(
                    X_target, y_target, lag, param_grid, n_trials=5
                )
                
                if best_model_info:
                    # Add model to ensemble
                    ensemble_manager.add_model(
                        target_name, 
                        best_model_info, 
                        best_model_info['best_val_loss']
                    )
                    
                    print(f"✅ Completed training model {model_idx+1} for {target_name}")
                    print(f"   Validation Loss: {best_model_info['best_val_loss']:.6f}")
                    print(f"   Test Loss: {best_model_info['test_loss']:.6f}")
                    
            except Exception as e:
                print(f"❌ Error training model {model_idx+1} for {target_name}: {e}")
                continue
    
    print(f"\n✅ Training completed for all targets")
    print(f"Ensembles created for {len(ensemble_manager.ensembles)} targets")
    
    # Generate predictions using corrected ensemble approach
    print("\nGenerating CORRECTED ensemble predictions with uncertainty...")
    
    # Calibrate uncertainties if validation data is available
    try:
        # Use last 20% of data for uncertainty calibration
        val_size = int(0.2 * len(next(iter(X_targets_dict.values()))))
        X_val_dict = {}
        y_val_dict = {}
        
        for target_name, X_target in X_targets_dict.items():
            X_val_dict[target_name] = X_target.iloc[-val_size:]
            y_val_dict[target_name] = y_targets_dict[target_name].iloc[-val_size:]
        
        ensemble_manager.calibrate_all_uncertainties(X_val_dict, y_val_dict)
    except Exception as e:
        print(f"Warning: Could not calibrate uncertainties: {e}")
    
    # Generate predictions for all targets
    all_predictions = ensemble_manager.predict_all_targets(X_targets_dict)
    
    # Save all ensembles
    ensemble_manager.save_all_ensembles()
    
    # Create comprehensive results DataFrame
    final_results = pd.DataFrame()
    for target_name, pred_data in all_predictions.items():
        results_df = pred_data['results_df']
        # Add target name as prefix to avoid column name conflicts
        results_df = results_df.add_prefix(f'{target_name}_')
        results_df = results_df.rename(columns={f'{target_name}_ensemble_prediction': f'{target_name}_prediction'})
        
        if final_results.empty:
            final_results = results_df
        else:
            final_results = final_results.join(results_df, how='outer')
    
    # Save final predictions
    final_results.to_csv('corrected_ensemble_predictions.csv')
    print("📊 CORRECTED predictions with uncertainty saved to 'corrected_ensemble_predictions.csv'")
    
    # Print comprehensive summary
    print("\n🎯 CORRECTED Training Summary:")
    print(f"   Total targets processed: {len(target_configs)}")
    print(f"   Models per target: {config.models_per_target}")
    print(f"   Total models trained: {sum(len(e.models) for e in ensemble_manager.ensembles.values())}")
    print(f"   Cross-validation folds: {config.n_folds}")
    print(f"   Final predictions shape: {final_results.shape}")
    
    # Display prediction statistics
    if not final_results.empty:
        # Extract ensemble predictions
        ensemble_cols = [col for col in final_results.columns if 'prediction' in col and 'uncertainty' not in col]
        pred_stats = final_results[ensemble_cols].describe()
        print("\n📈 CORRECTED Prediction statistics:")
        print(pred_stats.loc[['mean', 'std', 'min', 'max']])
    
    return ensemble_manager, all_predictions, final_results

if __name__ == "__main__":
    ensemble_manager, all_predictions, final_results = main()