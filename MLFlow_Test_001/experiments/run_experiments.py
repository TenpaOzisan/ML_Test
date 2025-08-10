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