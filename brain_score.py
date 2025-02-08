import numpy as np
import torch
from torch import nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RidgeRegression(nn.Module):
    """
    PyTorch implementation of Ridge Regression
    """
    def __init__(self, input_dim):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)
        
    def forward(self, x):
        return self.linear(x)
    
def fisher_z_transform(r):
    """
    Apply Fisher Z transformation to correlation coefficient
    """
    # Clip r to avoid numerical issues
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

def compute_correlation(y_true, y_pred):
    """
    Compute Pearson correlation and its Fisher Z transformation
    """
    # Convert to numpy if tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Compute correlation
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Apply Fisher Z transformation
    z_score = fisher_z_transform(correlation)
    
    return correlation, z_score

def nested_loocv_ridge_cuda(X_train, y_train, alphas, device, val_split=0.2):
    """
    Perform Ridge Regression for multiple alphas and find the best one.
    
    - Uses a single training and validation split.
    - Reduces computations from O(n_samples × n_alphas) to O(n_alphas).
    
    Parameters:
        X_train (ndarray): Training feature matrix (n_samples, n_features).
        y_train (ndarray): Target values (n_samples,).
        alphas (list): List of ridge regression regularization parameters.
        device (str): 'cuda' or 'cpu'.
        val_split (float): Proportion of data to use for validation (default 20%).
    
    Returns:
        float: Best alpha.
    """
    
    n_samples = len(y_train)
    split_idx = int(n_samples * (1 - val_split))  # Train-validation split index
    
    # Convert data to PyTorch tensors and move to GPU if available
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    
    # Split into training and validation sets
    X_train_split, X_val_split = X_train[:split_idx], X_train[split_idx:]
    y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]

    mse_alphas = torch.zeros(len(alphas), device=device)
    
    for j, alpha in enumerate(alphas):
        # Compute (X^T X + alpha*I)
        XtX = torch.mm(X_train_split.t(), X_train_split)
        reg_term = alpha * torch.eye(X_train_split.shape[1], device=device)  # Regularization term
        Xty = torch.mm(X_train_split.t(), y_train_split.unsqueeze(1))
        
        # Solve for ridge regression weights: (X^T X + alpha I)^(-1) X^T y
        weights = torch.linalg.solve(XtX + reg_term, Xty)
        
        # Compute predictions on validation set
        y_pred = torch.mm(X_val_split, weights)
        
        # Compute Mean Squared Error (MSE)
        mse = torch.nn.functional.mse_loss(y_pred, y_val_split.unsqueeze(1))
        mse_alphas[j] = mse
    
    # Select the best alpha based on lowest MSE
    best_alpha = alphas[torch.argmin(mse_alphas)]
    
    return best_alpha.item()

def time_series_ridge_cv_cuda(X, y, n_splits=5, device='cuda'):
    """
    GPU-accelerated ridge regression with nested cross-validation and correlation analysis
    """
    # Ensure y is 1D
    y = y.ravel()
    
    # Create logarithmically spaced alphas
    alphas = torch.logspace(-1, 8, 10).to(device)
    
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Initialize arrays to store results
    predictions = np.zeros_like(y)
    coefficients = []
    alphas_selected = []
    correlations = []
    z_scores = []
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Find best alpha using nested LOOCV
        best_alpha = nested_loocv_ridge_cuda(X_train, y_train, alphas, device)
        alphas_selected.append(best_alpha)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        # Solve ridge regression analytically on GPU
        XtX = torch.mm(X_train_tensor.t(), X_train_tensor)
        reg_term = best_alpha * torch.eye(X_train.shape[1], device=device)
        Xty = torch.mm(X_train_tensor.t(), y_train_tensor.unsqueeze(1))
        weights = torch.linalg.solve(XtX + reg_term, Xty)
        
        # Generate predictions
        fold_predictions = torch.mm(X_test_tensor, weights).cpu().numpy().ravel()
        predictions[test_idx] = fold_predictions
        coefficients.append(weights.cpu().numpy().ravel())
        
        # Compute correlation and Fisher Z score for this fold
        fold_corr, fold_z = compute_correlation(y_test, fold_predictions)
        correlations.append(fold_corr)
        z_scores.append(fold_z)
        
        #print(f"Fold {fold + 1}:")
        #print(f"  Selected alpha = {best_alpha:.2e}")
        #print(f"  Correlation = {fold_corr:.4f}")
        #print(f"  Fisher Z = {fold_z:.4f}")
    
    # Calculate average correlation and Z-score
    mean_correlation = np.mean(correlations)
    mean_z = np.mean(z_scores)
    std_z = np.std(z_scores, ddof=1)  # Sample standard deviation
    
    # Calculate standard error of mean Z
    se_z = std_z / np.sqrt(len(z_scores))
    
    results = {
        'predictions': predictions,
        'coefficients': coefficients,
        'alphas': alphas_selected,
        'correlations': correlations,
        'z_scores': z_scores,
        'mean_correlation': mean_correlation,
        'mean_z': mean_z,
        'std_z': std_z,
        'se_z': se_z
    }
    
    return results
