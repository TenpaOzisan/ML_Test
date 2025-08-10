"""データ生成モジュール"""
import numpy as np

def generate_single_feature_data(n_samples=100, random_seed=0):
    """
    単一特徴量（魚の数）のデータを生成
    
    Parameters:
    -----------
    n_samples : int
        生成するサンプル数
    random_seed : int
        乱数シード
    
    Returns:
    --------
    X : ndarray
        魚の数 (1-50匹)
    y : ndarray
        酸素濃度 (mg/L)
    """
    np.random.seed(random_seed)
    
    # 特徴量: 魚の数
    X = np.random.randint(1, 51, n_samples).reshape(-1, 1)
    
    # 目的変数: 酸素濃度
    y = 10 - 0.15 * X.flatten() + np.random.normal(0, 0.5, n_samples)
    
    return X, y

def generate_multi_feature_data(n_samples=150, random_seed=0):
    """
    複数特徴量（魚の数、水槽容量）のデータを生成
    
    Parameters:
    -----------
    n_samples : int
        生成するサンプル数
    random_seed : int
        乱数シード
    
    Returns:
    --------
    X : ndarray
        特徴量行列 [魚の数, 水槽容量]
    y : ndarray
        酸素濃度 (mg/L)
    """
    np.random.seed(random_seed)
    
    # 特徴量1: 魚の数 (1-50匹)
    X_fish_count = np.random.randint(1, 51, n_samples)
    
    # 特徴量2: 水槽容量 (30-120L)
    X_tank_capacity = np.random.randint(30, 121, n_samples)
    
    # 特徴量行列
    X = np.c_[X_fish_count, X_tank_capacity]
    
    # 目的変数: 酸素濃度
    y = 8 - 0.15 * X_fish_count + 0.05 * X_tank_capacity + np.random.normal(0, 0.5, n_samples)
    
    return X, y