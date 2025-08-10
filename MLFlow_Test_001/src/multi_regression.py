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