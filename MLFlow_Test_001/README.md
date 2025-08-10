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