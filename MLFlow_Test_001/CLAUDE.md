# MLFlow機械学習実験管理プロジェクト

## プロジェクト概要

このプロジェクトは、水槽内の魚の数と酸素濃度の関係を予測する機械学習モデルをMLFlowで管理します。単回帰分析と重回帰分析の両方の実験を追跡し、モデルのパフォーマンスを比較できるようにします。

## セットアップ

### 1. プロジェクトディレクトリの作成

```bash
mkdir mlflow-fish-oxygen-prediction
cd mlflow-fish-oxygen-prediction
```

### 2. 仮想環境の作成と有効化

```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
```

### 3. 必要なパッケージのインストール

```bash
pip install numpy scikit-learn matplotlib mlflow pandas
```

### 4. requirements.txtの作成

```bash
echo "numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
mlflow==2.8.0
pandas==2.0.3" > requirements.txt
```

## ファイル構成

```
mlflow-fish-oxygen-prediction/
├── CLAUDE.md                 # このファイル
├── requirements.txt          # 依存パッケージリスト
├── src/
│   ├── __init__.py
│   ├── data_generator.py    # データ生成モジュール
│   ├── single_regression.py # 単回帰分析
│   ├── multi_regression.py  # 重回帰分析
│   └── utils.py             # ユーティリティ関数
├── experiments/
│   └── run_experiments.py   # 実験実行スクリプト
├── mlruns/                  # MLFlow実験データ（自動生成）
└── README.md                # プロジェクト説明
```

## 実装コード

### src/__init__.py

```python
"""MLFlow Fish Oxygen Prediction Package"""
__version__ = "1.0.0"
```

### src/data_generator.py

```python
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
```

### src/utils.py

```python
"""ユーティリティ関数"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import mlflow

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
    
    plt.show()
    plt.close()

def plot_multi_regression(X_test, y_test, model, save_path=None):
    """重回帰の結果を3Dプロット"""
    fig = plt.figure(figsize=(12, 8))
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
    
    ax.set_xlabel('魚の数 (匹)')
    ax.set_ylabel('水槽容量 (L)')
    ax.set_zlabel('酸素濃度 (mg/L)')
    ax.set_title('重回帰モデルによる予測結果')
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(save_path)
    
    plt.show()
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
```

### src/single_regression.py

```python
"""単回帰分析の実装"""
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from .data_generator import generate_single_feature_data
from .utils import plot_single_regression, calculate_metrics

def run_single_regression_experiment(experiment_name="single_regression", run_name=None):
    """
    単回帰分析の実験を実行
    
    Parameters:
    -----------
    experiment_name : str
        MLFlow実験名
    run_name : str
        実行名
    
    Returns:
    --------
    model : LinearRegression
        学習済みモデル
    metrics : dict
        評価指標
    """
    # MLFlow実験の設定
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        # パラメータをログ
        params = {
            "model_type": "LinearRegression",
            "features": "fish_count",
            "test_size": 0.2,
            "random_state": 0,
            "n_samples": 100
        }
        mlflow.log_params(params)
        
        # データ生成
        X, y = generate_single_feature_data(n_samples=params["n_samples"])
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params["test_size"], random_state=params["random_state"]
        )
        
        # モデル学習
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 予測
        y_pred = model.predict(X_test)
        
        # メトリクスを計算
        metrics = calculate_metrics(y_test, y_pred)
        
        # モデル係数をログ
        mlflow.log_metric("coefficient", model.coef_[0])
        mlflow.log_metric("intercept", model.intercept_)
        
        # 評価指標をログ
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # プロットを保存
        plot_single_regression(
            X_train, y_train, X_test, y_test, model,
            save_path="single_regression_plot.png"
        )
        
        # モデルを保存
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=mlflow.models.infer_signature(X_train, y_train)
        )
        
        print(f"単回帰モデルの学習完了")
        print(f"係数: {model.coef_[0]:.4f}")
        print(f"切片: {model.intercept_:.4f}")
        print(f"R2スコア: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        
        return model, metrics
```

### src/multi_regression.py

```python
"""重回帰分析の実装"""
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from .data_generator import generate_multi_feature_data
from .utils import plot_multi_regression, calculate_metrics

def run_multi_regression_experiment(experiment_name="multi_regression", run_name=None):
    """
    重回帰分析の実験を実行
    
    Parameters:
    -----------
    experiment_name : str
        MLFlow実験名
    run_name : str
        実行名
    
    Returns:
    --------
    model : LinearRegression
        学習済みモデル
    metrics : dict
        評価指標
    """
    # MLFlow実験の設定
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        # パラメータをログ
        params = {
            "model_type": "LinearRegression",
            "features": "fish_count,tank_capacity",
            "test_size": 0.2,
            "random_state": 0,
            "n_samples": 150
        }
        mlflow.log_params(params)
        
        # データ生成
        X, y = generate_multi_feature_data(n_samples=params["n_samples"])
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params["test_size"], random_state=params["random_state"]
        )
        
        # モデル学習
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 予測
        y_pred = model.predict(X_test)
        
        # メトリクスを計算
        metrics = calculate_metrics(y_test, y_pred)
        
        # モデル係数をログ
        mlflow.log_metric("coefficient_fish_count", model.coef_[0])
        mlflow.log_metric("coefficient_tank_capacity", model.coef_[1])
        mlflow.log_metric("intercept", model.intercept_)
        
        # 評価指標をログ
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # プロットを保存
        plot_multi_regression(
            X_test, y_test, model,
            save_path="multi_regression_plot.png"
        )
        
        # モデルを保存
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=mlflow.models.infer_signature(X_train, y_train)
        )
        
        print(f"重回帰モデルの学習完了")
        print(f"係数 (魚の数): {model.coef_[0]:.4f}")
        print(f"係数 (水槽容量): {model.coef_[1]:.4f}")
        print(f"切片: {model.intercept_:.4f}")
        print(f"R2スコア: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        
        return model, metrics
```

### experiments/run_experiments.py

```python
"""実験実行スクリプト"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from src.single_regression import run_single_regression_experiment
from src.multi_regression import run_multi_regression_experiment

def main():
    """メイン実行関数"""
    
    # MLFlowトラッキングURIの設定（ローカル）
    mlflow.set_tracking_uri("file:./mlruns")
    
    print("=" * 50)
    print("MLFlow実験管理システム起動")
    print("=" * 50)
    
    # 1. 単回帰分析の実験
    print("\n[1] 単回帰分析の実験を開始...")
    single_model, single_metrics = run_single_regression_experiment(
        experiment_name="fish_oxygen_single_regression",
        run_name="baseline_single_feature"
    )
    
    # 2. 重回帰分析の実験
    print("\n[2] 重回帰分析の実験を開始...")
    multi_model, multi_metrics = run_multi_regression_experiment(
        experiment_name="fish_oxygen_multi_regression",
        run_name="baseline_multi_feature"
    )
    
    # 3. 結果の比較
    print("\n" + "=" * 50)
    print("実験結果の比較")
    print("=" * 50)
    
    print("\n単回帰モデル:")
    print(f"  R2スコア: {single_metrics['r2']:.4f}")
    print(f"  RMSE: {single_metrics['rmse']:.4f}")
    print(f"  MAE: {single_metrics['mae']:.4f}")
    
    print("\n重回帰モデル:")
    print(f"  R2スコア: {multi_metrics['r2']:.4f}")
    print(f"  RMSE: {multi_metrics['rmse']:.4f}")
    print(f"  MAE: {multi_metrics['mae']:.4f}")
    
    improvement = ((multi_metrics['r2'] - single_metrics['r2']) / single_metrics['r2']) * 100
    print(f"\n改善率 (R2スコア): {improvement:.2f}%")
    
    print("\n" + "=" * 50)
    print("MLFlow UIを起動するには以下のコマンドを実行:")
    print("mlflow ui --port 5000")
    print("ブラウザで http://localhost:5000 にアクセス")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

### README.md

```markdown
# 魚の数と酸素濃度予測モデル with MLFlow

## 概要

水槽内の環境データを用いて酸素濃度を予測する機械学習モデルの実験管理プロジェクト。
MLFlowを使用して実験の追跡、モデルの管理、メトリクスの比較を行います。

## 特徴

- **単回帰分析**: 魚の数から酸素濃度を予測
- **重回帰分析**: 魚の数と水槽容量から酸素濃度を予測
- **MLFlow統合**: 実験の自動追跡とモデル管理
- **可視化**: 予測結果の自動プロット生成

## インストール

```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

### 1. 実験の実行

```bash
python experiments/run_experiments.py
```

### 2. MLFlow UIの起動

```bash
mlflow ui --port 5000
```

ブラウザで http://localhost:5000 にアクセスして実験結果を確認。

### 3. 保存されたモデルの使用

```python
import mlflow

# モデルのロード
model_uri = "runs:/<RUN_ID>/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# 予測
import numpy as np
X_new = np.array([[25, 60]])  # 魚25匹、水槽60L
prediction = loaded_model.predict(X_new)
print(f"予測酸素濃度: {prediction[0]:.2f} mg/L")
```

## プロジェクト構成

```
.
├── src/                    # ソースコード
│   ├── data_generator.py   # データ生成
│   ├── single_regression.py # 単回帰
│   ├── multi_regression.py  # 重回帰
│   └── utils.py            # ユーティリティ
├── experiments/            # 実験スクリプト
├── mlruns/                # MLFlow実験データ
└── requirements.txt       # 依存パッケージ
```

## メトリクス

- **R2スコア**: 決定係数（モデルの説明力）
- **RMSE**: 二乗平均平方根誤差
- **MAE**: 平均絶対誤差
- **MSE**: 平均二乗誤差

## ライセンス

MIT License
```

## 実行手順

### 1. プロジェクトのセットアップ

```bash
# プロジェクトディレクトリを作成
mkdir mlflow-fish-oxygen-prediction
cd mlflow-fish-oxygen-prediction

# ファイル構造を作成
mkdir src experiments
touch src/__init__.py

# 仮想環境をセットアップ
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. コードファイルの作成

上記の各Pythonファイルを対応するディレクトリに作成してください。

### 3. 実験の実行

```bash
# 実験を実行
python experiments/run_experiments.py
```

### 4. MLFlow UIでの結果確認

```bash
# MLFlow UIを起動
mlflow ui --port 5000

# ブラウザで以下にアクセス
# http://localhost:5000
```

## MLFlowの主な機能

### 実験の追跡

- **パラメータ**: モデルの設定値（test_size、random_stateなど）
- **メトリクス**: 評価指標（R2、RMSE、MAEなど）
- **アーティファクト**: グラフ、モデルファイル

### モデルの管理

- モデルの保存と読み込み
- モデルのバージョン管理
- モデルの署名（入出力スキーマ）

### 実験の比較

- 複数の実験結果を並べて比較
- メトリクスの視覚的な比較
- パラメータとメトリクスの相関分析

## トラブルシューティング

### MLFlow UIが起動しない場合

```bash
# ポートを変更して試す
mlflow ui --port 5001

# または、既存のプロセスを終了
lsof -i :5000  # 使用中のプロセスを確認
kill -9 <PID>  # プロセスを終了
```

### グラフが表示されない場合

```bash
# matplotlibのバックエンドを確認
python -c "import matplotlib; print(matplotlib.get_backend())"

# 必要に応じてバックエンドを設定
export MPLBACKEND=Agg  # サーバー環境の場合
```

## 拡張案

1. **ハイパーパラメータチューニング**
   - GridSearchCVやOptunaとの統合
   - 自動的な最適パラメータ探索

2. **追加の特徴量**
   - 水温、pH値、照明時間などの追加
   - 特徴量の重要度分析

3. **異なるモデルの比較**
   - ランダムフォレスト、XGBoostなどの追加
   - アンサンブル学習の実装

4. **モデルのデプロイ**
   - MLFlow ModelsでのREST API化
   - Dockerコンテナ化

## 参考資料

- [MLFlow公式ドキュメント](https://mlflow.org/docs/latest/index.html)
- [scikit-learn公式ドキュメント](https://scikit-learn.org/stable/)
- [MLFlow Tracking](https://mlflow.org/docs/latest/tracking.html)