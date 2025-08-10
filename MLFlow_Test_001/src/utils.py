"""ユーティリティ関数"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import mlflow

# 日本語フォント設定
def setup_japanese_font():
    """日本語フォントを設定"""
    japanese_fonts = ['Hiragino Sans', 'YuGothic', 'Yu Gothic', 'Meiryo', 'Noto Sans CJK JP']
    for font in japanese_fonts:
        if font in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = font
            break
    else:
        # フォールバック: DejaVu Sans + 日本語サポート
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策

# フォント設定を初期化
setup_japanese_font()

def plot_single_regression(X_train, y_train, X_test, y_test, model, save_path=None):
    """単回帰の結果をプロット"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', label='訓練データ', alpha=0.5)
    plt.scatter(X_test, y_test, color='green', label='テストデータ', alpha=0.7)
    
    # 予測線
    X_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_pred_line = model.predict(X_range)
    plt.plot(X_range, y_pred_line, color='red', linewidth=2, label='予測線')
    
    plt.title('単回帰モデルによる予測結果')
    plt.xlabel('魚の数 (匹)')
    plt.ylabel('酸素濃度 (mg/L)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(save_path)
    
    plt.close()

def plot_multi_regression(X_test, y_test, model, save_path=None):
    """重回帰の結果を3Dプロット"""
    fig = plt.figure(figsize=(14, 10))  # サイズを大きく
    ax = fig.add_subplot(111, projection='3d')
    
    # テストデータをプロット
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, 
               c='green', marker='o', s=50, label='テストデータ', alpha=0.7)
    
    # 予測平面を作成
    x_surf = np.arange(0, 51, 2)
    y_surf = np.arange(30, 121, 5)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = model.predict(np.c_[x_surf.ravel(), y_surf.ravel()]).reshape(x_surf.shape)
    
    # 予測平面をプロット
    ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.3)
    
    # 軸ラベル設定（パディング調整）
    ax.set_xlabel('魚の数 (匹)', labelpad=10)
    ax.set_ylabel('水槽容量 (L)', labelpad=10)
    ax.set_zlabel('酸素濃度 (mg/L)', labelpad=15)  # Z軸は特に余裕を持たせる
    ax.set_title('重回帰モデルによる予測結果', pad=20)
    
    # 3Dプロットの表示角度を調整
    ax.view_init(elev=20, azim=45)
    
    # レイアウト調整
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.5)
        mlflow.log_artifact(save_path)
    
    plt.close()

def calculate_metrics(y_true, y_pred):
    """評価指標を計算"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }