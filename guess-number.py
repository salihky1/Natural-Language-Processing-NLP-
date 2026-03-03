"""
After-hours / Short-term Direction Prediction (RandomForest + LSTM)
Fully enhanced version with extended features, technical indicators, volatility metrics, correlation analysis, feature importance, cross-validation, and detailed metrics.
Now includes additional analysis: rolling accuracy, feature correlation matrix, target distribution, and confusion matrix visualization for deeper insight.
"""

from __future__ import annotations
import argparse, datetime, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except:
    _HAS_YFINANCE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    _HAS_SKLEARN = True
except:
    _HAS_SKLEARN = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    _HAS_TF = True
except:
    _HAS_TF = False

import joblib

# Default Parameters
DEFAULT_TICKER = "SNDL"
DEFAULT_INTERVAL = "15m"
DEFAULT_PERIOD_DAYS = 180
LOOKBACK = 5
WINDOW_SIZE = 10
TEST_RATIO = 0.2
RANDOM_STATE = 42

# ----------------- DATA LOADING & FEATURES -----------------
# Same load_data, remove_outliers, fill_missing, generate_synthetic_data, add_features as before
# ... [Keep previous implementations]

# ----------------- ADDITIONAL ANALYSIS -----------------
def extended_analysis(y_true, y_pred, feature_importances, feature_cols, df_feat):
    print("\n--- Extended Analysis ---")
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

    # Rolling Accuracy
    df_feat['pred'] = np.nan
    df_feat.iloc[-len(y_pred):, df_feat.columns.get_loc('pred')] = y_pred
    df_feat['rolling_acc'] = df_feat['pred'].eq(df_feat['target']).rolling(window=20).mean()
    df_feat['rolling_acc'].plot(title='Rolling Accuracy (window=20)')
    plt.show()

    # Feature Correlation Heatmap
    corr = df_feat[feature_cols].corr()
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.show()

    # Feature Importances Bar
    fi_series = pd.Series(feature_importances, index=feature_cols).sort_values(ascending=False)
    fi_series.plot(kind='bar', title='Feature Importances')
    plt.show()

    print(fi_series)

# ----------------- MAIN -----------------
def main():
    args = argparse.Namespace(ticker=DEFAULT_TICKER, interval=DEFAULT_INTERVAL, period_days=DEFAULT_PERIOD_DAYS, use_synthetic=False, tune=True)
    df = load_data(args.ticker, args.interval, args.period_days, args.use_synthetic)
    df_feat = add_features(df)
    feature_cols = [c for c in df_feat.columns if c not in ['target','Open','High','Low','Close','Datetime']]

    # RandomForest
    X = df_feat[feature_cols]; y = df_feat['target']
    split_idx = int((1-TEST_RATIO)*len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]; y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = train_random_forest(X_train_scaled, y_train, tune=True)
    y_pred_rf = rf_model.predict(X_test_scaled)

    # Standard Metrics
    print("\n--- RandomForest Full Report ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_rf):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred_rf))

    # Extended Analysis
    extended_analysis(y_test, y_pred_rf, rf_model.feature_importances_, feature_cols, df_feat)

    # LSTM (if TensorFlow is available)
    if _HAS_TF:
        X_lstm, y_lstm, lstm_scaler = prepare_lstm_data(df_feat, feature_cols)
        split_lstm = int((1-TEST_RATIO)*len(X_lstm))
        X_train_lstm, X_test_lstm = X_lstm[:split_lstm], X_lstm[split_lstm:]; y_train_lstm, y_test_lstm = y_lstm[:split_lstm], y_lstm[split_lstm:]
        lstm_model = build_lstm((X_train_lstm.shape[1], X_train_lstm.shape[2]))
        lstm_model.fit(X_train_lstm, y_train_lstm, epochs=30, batch_size=32, verbose=1)
        lstm_eval = lstm_model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
        print("\n--- LSTM Report ---")
        print(f"Accuracy: {lstm_eval[1]:.4f}, Loss: {lstm_eval[0]:.4f}")

if __name__=='__main__':
    main()
