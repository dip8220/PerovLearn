import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# === Load Dataset ===
data = pd.read_csv('D:/Final Year Project/Independent_expanded_with_Actual_Bandgap.csv')
X = data[['MA', 'FA', 'Cs', 'Cl', 'Br', 'I', 'Pb', 'Sn']]
y = data['Bandgap']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Feature Scaling ===
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# === Deep Models Definitions ===
def build_mlp():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_transformer_like():
    input_layer = Input(shape=(X_train.shape[1],))
    x = Dense(64, activation='relu')(input_layer)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(), loss='mse')
    return model

# === Train Base Models ===
ridge = Ridge().fit(X_train, y_train)
svr = SVR().fit(X_train, y_train)
bagging = BaggingRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)

mlp = build_mlp()
mlp.fit(X_train, y_train_scaled, epochs=100, batch_size=32, verbose=0)
mlp_preds = scaler_y.inverse_transform(mlp.predict(X_test)).ravel()

transformer = build_transformer_like()
transformer.fit(X_train, y_train_scaled, epochs=100, batch_size=32, verbose=0)
trans_preds = scaler_y.inverse_transform(transformer.predict(X_test)).ravel()

# === Predictions from Base Models ===
ridge_preds = ridge.predict(X_test)
svr_preds = svr.predict(X_test)
bag_preds = bagging.predict(X_test)
knn_preds = knn.predict(X_test)

# === Stack Predictions for Meta Learners ===
meta_X = np.column_stack((ridge_preds, svr_preds, bag_preds, knn_preds, mlp_preds, trans_preds))

# --- Blended Model (Linear Regression on y_test directly) ---
blended_model = LinearRegression()
blended_model.fit(meta_X, y_test)
blended_preds = np.mean(meta_X, axis=1)

# --- Stacked Model (Linear Regression as meta-learner) ---
stacked_lr_model = LinearRegression()
stacked_lr_model.fit(meta_X, y_test)
stacked_lr_preds = stacked_lr_model.predict(meta_X)

# --- Stacked Model (Random Forest as meta-learner) ---
stacked_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
stacked_rf_model.fit(meta_X, y_test)
stacked_rf_preds = stacked_rf_model.predict(meta_X)

performance = {}
y_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.delaxes(axes[1, 1])  # Remove empty subplot
axes = axes.flatten()

subplot_labels = ['(a)', '(b)', '(c)']

# === Evaluation Function ===
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    spearman = spearmanr(y_true, y_pred)[0]
    performance[name] = {'RMSE': rmse, 'R²': r2, 'Pearson': pearson, 'Spearman': spearman}
    #print(f"\n{name}:\nRMSE: {rmse:.6f}, R²: {r2:.6f}, Pearson: {pearson:.6f}, Spearman: {spearman:.6f}")
    #return rmse, r2, pearson, spearman

evaluate_model("Blended (LR)", y_test, blended_preds)
evaluate_model("Stacked (LR)", y_test, stacked_lr_preds)
evaluate_model("Stacked (RF)", y_test, stacked_rf_preds)

# === Filter Function for Visualization ===
def filter_close_points(x, y, tolerance=0.02):
    unique = []
    for xi, yi in zip(x, y):
        if not any(np.linalg.norm(np.array([xi, yi]) - np.array(p)) < tolerance for p in unique):
            unique.append((xi, yi))
    return np.array(unique)

# === Filtered Points for Each Model ===
filtered_blend = filter_close_points(y_test.values, blended_preds)
filtered_stack_lr = filter_close_points(y_test.values, stacked_lr_preds)
filtered_stack_rf = filter_close_points(y_test.values, stacked_rf_preds)

filtered_preds = [filtered_blend, filtered_stack_lr, filtered_stack_rf]

for i, filtered_pred in enumerate(filtered_preds):
    ax = axes[i]
    ax.scatter(filtered_pred[:, 0], filtered_pred[:, 1], alpha=0.6, color='b', label='Predicted', marker='o')
    ax.scatter(filtered_pred[:, 0], filtered_pred[:, 0], alpha=0.6, color='r', label='Actual', marker='^')
    ax.plot([1, 2.75], [1, 2.75], 'k--', label='Ideal Fit')
    ax.set_xlim(1, 2.75)
    ax.set_ylim(1, 2.75)
    ax.set_xlabel('Actual Bandgap', fontsize=12, fontfamily='sans-serif')
    ax.set_ylabel('Predicted Bandgap', fontsize=12, fontfamily='sans-serif')
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.text(0.5, -0.25, f'{subplot_labels[i]}', transform=ax.transAxes,
            fontsize=10, fontfamily='sans-serif', ha='center', va='top')

plt.tight_layout(pad=1.5)
plt.savefig('Merged_Ensemble_Comparison.tiff', dpi=1200, format='tiff')
plt.show()

# Print metrics
print("\nModel Performance Metrics:\n")
for model, metrics in performance.items():
    print(f"{model}: RMSE = {metrics['RMSE']}, R² = {metrics['R²']}, Pearson = {metrics['Pearson']}, Spearman = {metrics['Spearman']}")

# Heatmap
performance_df = pd.DataFrame.from_dict(performance, orient='index')
plt.figure(figsize=(8, 5))
sns.heatmap(performance_df, annot=True, cmap='YlOrBr', fmt='.6f', linewidth=.5, linecolor='#D3D3D3', cbar_kws={"shrink": 0.8})
plt.title('Model Performance Comparison')
plt.savefig('Merged_Ensemble_Comparison_Heatmap.tiff', format='tiff', dpi=1200)
plt.show()
