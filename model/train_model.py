"""
==============================================================================
AI-Driven Smart Pricing System for Theme Parks
Model Training Module
==============================================================================

Trains and compares four regression models for ticket price prediction.
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Install via: pip install xgboost")
    print("   Falling back to Gradient Boosting from sklearn.\n")
    from sklearn.ensemble import GradientBoostingRegressor


def load_and_prepare_data():
    """Load the dataset and prepare features for training."""
    print("=" * 60)
    print("  AI Smart Pricing System - Model Training")
    print("  Theme Park Revenue Optimizer")
    print("=" * 60)
    print()
    
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'theme_park_data.csv')
    data_path = os.path.abspath(data_path)
    
    print(f"📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Loaded {len(df)} records")
    print()
    
    # Feature Engineering
    print("🔧 Engineering features...")
    
    # Encode weather
    le_weather = LabelEncoder()
    df['weather_encoded'] = le_weather.fit_transform(df['weather'])
    
    # Add month and season features
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # Save encoder for later use
    encoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weather_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(le_weather, f)
    
    print(f"   Weather classes: {list(le_weather.classes_)}")
    print(f"   Weather encoding: {dict(zip(le_weather.classes_, le_weather.transform(le_weather.classes_)))}")
    print()
    
    return df, le_weather


def train_models(df):
    """Train four regression models on the same split and compare results."""
    
    # Define features and target
    feature_columns = [
        'day_of_week', 'is_weekend', 'is_holiday', 'weather_encoded',
        'temperature', 'competitor_price', 'visitors', 'month', 'quarter'
    ]
    target_column = 'recommended_price'
    
    X = df[feature_columns]
    y = df[target_column]
    
    print(f"📊 Features: {feature_columns}")
    print(f"🎯 Target: {target_column}")
    print(f"📏 Dataset shape: {X.shape}")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"🔀 Train set: {len(X_train)} samples")
    print(f"🔀 Test set:  {len(X_test)} samples")
    print()
    
    if HAS_XGBOOST:
        xgb_model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    else:
        xgb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    models = {
        'xgboost': xgb_model,
        'random_forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        'linear_regression': make_pipeline(StandardScaler(), LinearRegression()),
        'extra_trees': ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }
    display_names = {
        'xgboost': "XGBoost" if HAS_XGBOOST else "Gradient Boosting",
        'random_forest': 'Random Forest',
        'linear_regression': 'Linear Regression',
        'extra_trees': 'Extra Trees',
    }
    results = {}
    predictions = {}

    for name, model in models.items():
        print("━" * 50)
        print(f"🚀 Training {display_names[name]} Regressor")
        print("━" * 50)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        predictions[name] = y_pred
        results[name] = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
        }

        metrics = results[name]
        print(f"\n📈 {display_names[name]} Results:")
        print(f"   • R² Score:     {metrics['r2']:.4f} ({metrics['r2'] * 100:.2f}%)")
        print(f"   • MAE:          ₹{metrics['mae']:.2f}")
        print(f"   • RMSE:         ₹{metrics['rmse']:.2f}")
        print(f"   • CV R² Mean:   {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")

    comparison_df = pd.DataFrame({
        display_names[name]: [
            f"{metrics['r2']:.4f}",
            f"₹{metrics['mae']:.2f}",
            f"₹{metrics['rmse']:.2f}",
            f"{metrics['cv_mean']:.4f}",
        ]
        for name, metrics in results.items()
    }, index=['R² Score', 'MAE (₹)', 'RMSE (₹)', 'CV R² Mean'])
    print("\n⚖️  Model Comparison")
    print(comparison_df.to_string())

    winner = max(results, key=lambda name: results[name]['r2'])
    print(f"\n🏆 Winner: {display_names[winner]} (highest holdout R²: {results[winner]['r2']:.4f})\n")

    return models, results, winner, feature_columns, X_test, y_test, predictions


def save_models(models, results, winner, feature_columns):
    """Save the trained models and metadata."""
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save the best model as the primary model
    best_model = models[winner]
    model_path = os.path.join(model_dir, 'pricing_model.pkl')
    
    model_package = {
        'model': best_model,
        'model_name': winner,
        'feature_columns': feature_columns,
        'results': results,
        'winner': winner
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"💾 Best model saved to: {model_path}")
    
    # Also save both models individually
    for name, model in models.items():
        path = os.path.join(model_dir, f'{name}_model.pkl')
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 {name} model saved to: {path}")
    
    # Save results for dashboard
    results_path = os.path.join(model_dir, 'model_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"💾 Results saved to: {results_path}")
    print()


def plot_feature_importance(models, feature_columns):
    """Generate and save feature importance charts."""
    model_dir = os.path.dirname(os.path.abspath(__file__))

    importance_models = {
        name: model for name, model in models.items()
        if hasattr(model, 'feature_importances_')
    }
    fig, axes = plt.subplots(1, len(importance_models), figsize=(8 * len(importance_models), 6), squeeze=False)
    fig.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')

    for idx, (name, model) in enumerate(importance_models.items()):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_columns)))
        
        axes[0][idx].barh(
            feature_importance_df['Feature'],
            feature_importance_df['Importance'],
            color=colors
        )
        display_name = name.replace('_', ' ').title()
        if name == 'xgboost' and not HAS_XGBOOST:
            display_name = 'Gradient Boosting'
        axes[0][idx].set_title(f'{display_name}', fontsize=13)
        axes[0][idx].set_xlabel('Importance Score')
    
    plt.tight_layout()
    chart_path = os.path.join(model_dir, 'feature_importance.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Feature importance chart saved to: {chart_path}")


def plot_predictions(y_test, predictions):
    """Plot actual vs predicted prices for all models."""
    model_dir = os.path.dirname(os.path.abspath(__file__))

    ncols = 2
    nrows = int(np.ceil(len(predictions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows), squeeze=False)
    fig.suptitle('Actual vs Predicted Prices', fontsize=16, fontweight='bold')

    display_names = {
        'xgboost': "XGBoost" if HAS_XGBOOST else "Gradient Boosting",
        'random_forest': 'Random Forest',
        'linear_regression': 'Linear Regression',
        'extra_trees': 'Extra Trees',
    }
    for idx, (name, preds) in enumerate(predictions.items()):
        ax = axes[idx // ncols][idx % ncols]
        ax.scatter(y_test, preds, alpha=0.4, s=10, color='#2ecc71')
        ax.plot(
            [y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', lw=2, label='Perfect Prediction'
        )
        ax.set_xlabel('Actual Price (₹)')
        ax.set_ylabel('Predicted Price (₹)')
        ax.set_title(display_names[name])
        ax.legend()
    for idx in range(len(predictions), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    chart_path = os.path.join(model_dir, 'prediction_comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Prediction comparison chart saved to: {chart_path}")


def main():
    """Main training pipeline."""
    # Load data
    df, le_weather = load_and_prepare_data()
    
    # Train models
    models, results, winner, feature_columns, X_test, y_test, predictions = train_models(df)
    
    # Save models
    save_models(models, results, winner, feature_columns)
    
    # Generate visualizations
    print("📊 Generating visualizations...")
    plot_feature_importance(models, feature_columns)
    plot_predictions(y_test, predictions)
    
    print()
    print("=" * 60)
    print("  ✅ Model training pipeline completed successfully!")
    print(f"  🏆 Best model: {winner.replace('_', ' ').title()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
