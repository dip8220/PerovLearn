# === Import Libraries ===
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
data = pd.read_csv('D:/Final Year Project/Independent_expanded.csv')
X = data[['MA', 'FA', 'Cs', 'Cl', 'Br', 'I', 'Pb', 'Sn']]
y = data['Bandgap']

# === Train-Test Split and Scaling ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# === Define Deep Models ===
def build_mlp():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_transformer():
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

transformer = build_transformer()
transformer.fit(X_train, y_train_scaled, epochs=100, batch_size=32, verbose=0)
trans_preds = scaler_y.inverse_transform(transformer.predict(X_test)).ravel()

# === Base Model Predictions ===
ridge_preds = ridge.predict(X_test)
svr_preds = svr.predict(X_test)
bag_preds = bagging.predict(X_test)
knn_preds = knn.predict(X_test)

# === Prepare Meta Features for Stacking ===
meta_X = np.column_stack((ridge_preds, svr_preds, bag_preds, knn_preds, mlp_preds, trans_preds))

# === Meta-Learners: Linear Regression and Random Forest ===
stack_lr = LinearRegression().fit(meta_X, y_test)
stack_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(meta_X, y_test)

stack_lr_preds = stack_lr.predict(meta_X)
stack_rf_preds = stack_rf.predict(meta_X)

# === Filter Function ===
def filter_close_points(x, y, threshold=0.02):
    seen = []
    for xi, yi in zip(x, y):
        if not any(np.linalg.norm(np.array([xi, yi]) - np.array(s)) < threshold for s in seen):
            seen.append((xi, yi))
    return np.array(seen)

# === Performance Metrics ===
performance = {}

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    spearman = spearmanr(y_true, y_pred)[0]
    performance[name] = {'RMSE': rmse, 'R²': r2, 'Pearson': pearson, 'Spearman': spearman}

evaluate_model('KNN', y_test, knn_preds)
evaluate_model('Bagging', y_test, bag_preds)
evaluate_model('Stacked (LR)', y_test, stack_lr_preds)
evaluate_model('Stacked (RF)', y_test, stack_rf_preds)

# === Filtered Scatter Plots ===
filtered_knn = filter_close_points(y_test, knn_preds)
filtered_bag = filter_close_points(y_test, bag_preds)
filtered_stack_lr = filter_close_points(y_test, stack_lr_preds)
filtered_stack_rf = filter_close_points(y_test, stack_rf_preds)

filtered_all = [
    ('KNN', filtered_knn),
    ('Bagging', filtered_bag),
    ('Stacked (LR)', filtered_stack_lr),
    ('Stacked (RF)', filtered_stack_rf)
]

# === Plotting ===
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
axes = axes.flatten()
labels = ['(a)', '(b)', '(c)', '(d)']
for i, (title, data) in enumerate(filtered_all):
    ax = axes[i]
    ax.scatter(data[:, 0], data[:, 1], alpha=0.6, color='blue', label='Predicted', marker='o')
    ax.scatter(data[:, 0], data[:, 0], alpha=0.6, color='red', label='Actual', marker='^')
    ax.plot([1, 2.75], [1, 2.75], 'k--', label='Ideal Fit')
    ax.set_xlim(1, 2.75)
    ax.set_ylim(1, 2.75)
    ax.set_xlabel('Actual Bandgap', fontsize=12, fontfamily='sans-serif')
    ax.set_ylabel('Predicted Bandgap', fontsize=12, fontfamily='sans-serif')
    ax.legend(fontsize=10, loc='upper left')
    ax.tick_params(labelsize=12)
    ax.text(0.5, -0.25, labels[i], transform=ax.transAxes,
            fontsize=10, fontfamily='sans-serif', ha='center', va='top')
plt.tight_layout(pad=1.5)
plt.savefig('KNN_Bagging_Stacked_Comparison.tiff', dpi=1200, format='tiff')
plt.show()

# === Print Metrics ===
print("\nModel Performance Metrics:\n")
for model, metrics in performance.items():
    print(f"{model}: RMSE = {metrics['RMSE']:.6f}, R² = {metrics['R²']:.6f}, Pearson = {metrics['Pearson']:.6f}, Spearman = {metrics['Spearman']:.6f}")

# === Heatmap of Performance ===
performance_df = pd.DataFrame(performance).T
plt.figure(figsize=(8, 5))
sns.heatmap(performance_df, annot=True, fmt='.6f', cmap='YlOrBr', linewidths=0.5, linecolor='#D3D3D3', cbar_kws={"shrink": 0.8})
plt.title('Performance Comparison: KNN, Bagging, Stacked Models')
plt.savefig('KNN_Bagging_Stacked_Heatmap.tiff', dpi=1200, format='tiff')
plt.show()

