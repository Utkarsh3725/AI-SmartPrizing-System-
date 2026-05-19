"""
==============================================================================
AI-Driven Smart Pricing System for Theme Parks
Model Training Module
Team B | Utkarsh Arya | SRM University | Deloitte Digital Camp 2026
==============================================================================

Trains and compares XGBoost and Random Forest regression models
for optimal ticket price prediction.
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
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
    print("  Team B | Utkarsh Arya | SRM University")
    print("  Deloitte Digital Camp 2026")
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
    """Train XGBoost and Random Forest models and compare them."""
    
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
    
    results = {}
    models = {}
    
    # ========================
    # Model 1: XGBoost
    # ========================
    print("━" * 50)
    print("🚀 Training Model 1: XGBoost Regressor")
    print("━" * 50)
    
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
    
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    
    # Cross-validation
    xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')
    
    model_name = "XGBoost" if HAS_XGBOOST else "GradientBoosting"
    print(f"\n📈 {model_name} Results:")
    print(f"   • R² Score:     {xgb_r2:.4f} ({xgb_r2*100:.2f}%)")
    print(f"   • MAE:          ₹{xgb_mae:.2f}")
    print(f"   • RMSE:         ₹{xgb_rmse:.2f}")
    print(f"   • CV R² Mean:   {xgb_cv_scores.mean():.4f} (±{xgb_cv_scores.std():.4f})")
    print()
    
    results['xgboost'] = {
        'r2': xgb_r2, 'mae': xgb_mae, 'rmse': xgb_rmse,
        'cv_mean': xgb_cv_scores.mean(), 'cv_std': xgb_cv_scores.std()
    }
    models['xgboost'] = xgb_model
    
    # ========================
    # Model 2: Random Forest
    # ========================
    print("━" * 50)
    print("🌲 Training Model 2: Random Forest Regressor")
    print("━" * 50)
    
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    
    rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    
    print(f"\n📈 Random Forest Results:")
    print(f"   • R² Score:     {rf_r2:.4f} ({rf_r2*100:.2f}%)")
    print(f"   • MAE:          ₹{rf_mae:.2f}")
    print(f"   • RMSE:         ₹{rf_rmse:.2f}")
    print(f"   • CV R² Mean:   {rf_cv_scores.mean():.4f} (±{rf_cv_scores.std():.4f})")
    print()
    
    results['random_forest'] = {
        'r2': rf_r2, 'mae': rf_mae, 'rmse': rf_rmse,
        'cv_mean': rf_cv_scores.mean(), 'cv_std': rf_cv_scores.std()
    }
    models['random_forest'] = rf_model
    
    # ========================
    # Comparison
    # ========================
    print("━" * 50)
    print("⚖️  Model Comparison")
    print("━" * 50)
    
    comparison_df = pd.DataFrame({
        'Metric': ['R² Score', 'MAE (₹)', 'RMSE (₹)', 'CV R² Mean'],
        model_name: [
            f"{xgb_r2:.4f}", f"₹{xgb_mae:.2f}", f"₹{xgb_rmse:.2f}", f"{xgb_cv_scores.mean():.4f}"
        ],
        'Random Forest': [
            f"{rf_r2:.4f}", f"₹{rf_mae:.2f}", f"₹{rf_rmse:.2f}", f"{rf_cv_scores.mean():.4f}"
        ]
    })
    print()
    print(comparison_df.to_string(index=False))
    print()
    
    # Determine winner
    if xgb_r2 > rf_r2:
        winner = 'xgboost'
        winner_name = model_name
        print(f"🏆 Winner: {model_name} (Higher R² Score: {xgb_r2:.4f} vs {rf_r2:.4f})")
    else:
        winner = 'random_forest'
        winner_name = 'Random Forest'
        print(f"🏆 Winner: Random Forest (Higher R² Score: {rf_r2:.4f} vs {xgb_r2:.4f})")
    
    print()
    
    return models, results, winner, feature_columns, X_test, y_test, xgb_pred, rf_pred


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
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')
    
    # Plot for each model
    for idx, (name, model) in enumerate(models.items()):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_columns)))
        
        axes[idx].barh(
            feature_importance_df['Feature'],
            feature_importance_df['Importance'],
            color=colors
        )
        display_name = name.replace('_', ' ').title()
        if name == 'xgboost' and not HAS_XGBOOST:
            display_name = 'Gradient Boosting'
        axes[idx].set_title(f'{display_name}', fontsize=13)
        axes[idx].set_xlabel('Importance Score')
    
    plt.tight_layout()
    chart_path = os.path.join(model_dir, 'feature_importance.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Feature importance chart saved to: {chart_path}")


def plot_predictions(y_test, xgb_pred, rf_pred):
    """Plot actual vs predicted prices for both models."""
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Actual vs Predicted Prices', fontsize=16, fontweight='bold')
    
    model_name = "XGBoost" if HAS_XGBOOST else "GradientBoosting"
    
    for idx, (name, preds) in enumerate([(model_name, xgb_pred), ('Random Forest', rf_pred)]):
        axes[idx].scatter(y_test, preds, alpha=0.4, s=10, color='#2ecc71')
        axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                      'r--', lw=2, label='Perfect Prediction')
        axes[idx].set_xlabel('Actual Price (₹)')
        axes[idx].set_ylabel('Predicted Price (₹)')
        axes[idx].set_title(f'{name}')
        axes[idx].legend()
    
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
    models, results, winner, feature_columns, X_test, y_test, xgb_pred, rf_pred = train_models(df)
    
    # Save models
    save_models(models, results, winner, feature_columns)
    
    # Generate visualizations
    print("📊 Generating visualizations...")
    plot_feature_importance(models, feature_columns)
    plot_predictions(y_test, xgb_pred, rf_pred)
    
    print()
    print("=" * 60)
    print("  ✅ Model training pipeline completed successfully!")
    print(f"  🏆 Best model: {winner.replace('_', ' ').title()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
