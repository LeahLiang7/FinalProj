"""
Simple XGBoost with ALL features from filtered_energy_data.csv
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
df = pd.read_csv('filtered_energy_data.csv')
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Extract hour of day (0-23) from Time column
df['Time'] = pd.to_datetime(df['Time'])
df['hour'] = df['Time'].dt.hour  # 0-23, cycles across 3 days

# Prepare features and target
# Exclude: Time (datetime), BS (identifier), CellName (identifier), Energy (target)
# Include all other columns as features
# Use 'hour' (0-23) instead of 'Hours' (0-72) for better periodicity
feature_cols = ['load', 'ESMode1', 'ESMode2', 'ESMode3', 'ESMode4', 'ESMode5', 'ESMode6',
                'hour', 'RUType', 'Mode', 'Frequency', 'Bandwidth', 'Antennas', 'TXpower', 'BS_num']

X = df[feature_cols].copy()
y = df['Energy'].values

print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

# Encode categorical columns (RUType, Mode if any)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    print(f"After encoding, feature count: {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Scale features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Train XGBoost
print("\nTraining XGBoost...")
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_scaled, y_train, verbose=False)

# Predictions
y_train_pred = xgb_model.predict(X_train_scaled)
y_test_pred = xgb_model.predict(X_test_scaled)

# Calculate metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

# Print results
print("\n" + "="*60)
print("XGBoost Model Results (ALL Features)")
print("="*60)
print(f"\nNumber of features used: {X_train.shape[1]}")
print(f"\nTraining Set:")
print(f"  MAE:  {train_mae:.2f} kW")
print(f"  RMSE: {train_rmse:.2f} kW")
print(f"  MAPE: {train_mape:.2%}")

print(f"\nTest Set:")
print(f"  MAE:  {test_mae:.2f} kW")
print(f"  RMSE: {test_rmse:.2f} kW")
print(f"  MAPE: {test_mape:.2%}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Visualize top 10 feature importance
print("\nGenerating feature importance plot...")
top_10 = feature_importance.head(10)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(top_10)), top_10['importance'].values, color='steelblue', alpha=0.8)
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10['feature'].values)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Top 10 Most Important Features (XGBoost - All Features)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()  # Highest importance at top

plt.tight_layout()
plt.savefig('xgboost_all_features_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: xgboost_all_features_importance.png")

# Save results
with open('xgboost_all_features_results.txt', 'w', encoding='utf-8') as f:
    f.write("XGBoost Model Results (ALL Features)\n")
    f.write("="*60 + "\n\n")
    f.write(f"Number of features used: {X_train.shape[1]}\n\n")
    f.write(f"Training Set:\n")
    f.write(f"  MAE:  {train_mae:.2f} kW\n")
    f.write(f"  RMSE: {train_rmse:.2f} kW\n")
    f.write(f"  MAPE: {train_mape:.2%}\n\n")
    f.write(f"Test Set:\n")
    f.write(f"  MAE:  {test_mae:.2f} kW\n")
    f.write(f"  RMSE: {test_rmse:.2f} kW\n")
    f.write(f"  MAPE: {test_mape:.2%}\n\n")
    f.write(f"Top 10 Most Important Features:\n")
    f.write(feature_importance.head(10).to_string(index=False))

print("\nResults saved to: xgboost_all_features_results.txt")
