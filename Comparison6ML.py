import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers

# Load dataset
data = pd.read_csv('D:/Final Year Project/Independent_expanded.csv')
X = data[['MA', 'FA', 'Cs', 'Cl', 'Br', 'I', 'Pb', 'Sn']]
y = data['Bandgap']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# Filter function to remove closely packed points
def filter_close_points(x, y, threshold=0.01):
    filtered_x, filtered_y = [], []
    seen = set()
    for xi, yi in zip(x, y):
        key = (round(xi / threshold), round(yi / threshold))
        if key not in seen:
            seen.add(key)
            filtered_x.append(xi)
            filtered_y.append(yi)
    return np.array(filtered_x), np.array(filtered_y)

# Model functions
def get_ridge(): return Ridge()
def get_svr(): return SVR()
def get_bagging(): return BaggingRegressor()
def get_knn(): return KNeighborsRegressor()
def get_mlp(): return MLPRegressor(hidden_layer_sizes=(64, 32))
def get_transformer():
    input_layer = Input(shape=(X_train.shape[1],))
    x = layers.Dense(64, activation='relu')(input_layer)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# Fit + Predict function
def fit_predict(model, model_type='traditional'):
    if model_type == 'traditional':
        model.fit(X_train, y_train_scaled)
        y_pred_scaled = model.predict(X_test)
    elif model_type == 'dl':
        model.fit(X_train, y_train_scaled, epochs=100, batch_size=32, verbose=0)
        y_pred_scaled = model.predict(X_test).ravel()
    return scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Define model dictionary
models = {
    'Ridge': get_ridge,
    'SVR': get_svr,
    'Bagging': get_bagging,
    'KNN': get_knn,
    'MLP': get_mlp,
    'Transformer': get_transformer
}

performance = {}
y_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

# Individual Subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

for i, (name, model_func) in enumerate(models.items()):
    model = model_func()
    model_type = 'dl' if name == 'Transformer' else 'traditional'
    y_pred = fit_predict(model, model_type)

    # Metrics
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 6)
    r2 = round(r2_score(y_true, y_pred), 6)
    pearson = round(pearsonr(y_true, y_pred)[0], 6)
    spearman = round(spearmanr(y_true, y_pred)[0], 6)
    performance[name] = {'RMSE': rmse, 'R²': r2, 'Pearson': pearson, 'Spearman': spearman}

    # Filter sample points
    sample_idx = np.random.choice(len(y_true), size=200, replace=False)
    x_sampled = y_true[sample_idx]
    y_sampled = y_pred[sample_idx]
    filtered_y_test, filtered_y_pred = filter_close_points(x_sampled, y_sampled)

    # Plot
    ax = axes[i]
    ax.scatter(filtered_y_test, filtered_y_pred, alpha=0.6, color='b', label='Predicted', marker='o')
    ax.scatter(filtered_y_test, filtered_y_test, alpha=0.6, color='r', label='Actual', marker='^')
    ax.plot([1, 2.75], [1, 2.75], 'k--', label='Ideal Fit')
    ax.set_xlim(1, 2.75)
    ax.set_ylim(1, 2.75)
    ax.set_xlabel('Actual Bandgap', fontsize=12, fontfamily='sans-serif')
    ax.set_ylabel('Predicted Bandgap', fontsize=12, fontfamily='sans-serif')
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.text(0.5, -0.25, subplot_labels[i], transform=ax.transAxes,
            fontsize=10, fontfamily='sans-serif', ha='center', va='top')

plt.tight_layout(pad=1.5)
plt.savefig('Best_Performing_Models_Individual_Scatter.tiff', format='tiff', dpi=1200)
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
plt.savefig('Model_Performance_Comparison_Heatmap.tiff', format='tiff', dpi=1200)
plt.show()

