import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# Load dataset
data = pd.read_csv(r'D:\Final Year Project\Independent_expanded.csv')
X = data[['MA', 'FA', 'Cs', 'Cl', 'Br', 'I', 'Pb', 'Sn']]
y = data['Bandgap']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# Polynomial and Kernel Models
models = {
    'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
    'Kernel Ridge': KernelRidge()
}

performance = {}
sample_indices = np.random.choice(len(y_test), size=200, replace=False)

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.delaxes(axes[1, 1])  # Remove empty subplot
axes = axes.flatten()

subplot_labels = ['(a)', '(b)', '(c)']

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train_scaled)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

    filtered_y_test = y_true[sample_indices]
    filtered_y_pred = y_pred[sample_indices]

    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 6)
    r2 = round(r2_score(y_true, y_pred), 6)
    pearson = round(pearsonr(y_true, y_pred)[0], 6)
    spearman = round(spearmanr(y_true, y_pred)[0], 6)

    performance[name] = {
        'RMSE': rmse,
        'R²': r2,
        'Pearson': pearson,
        'Spearman': spearman
    }

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
    ax.text(0.5, -0.25, f'{subplot_labels[i]}', transform=ax.transAxes,
            fontsize=10, fontfamily='sans-serif', ha='center', va='top')

plt.tight_layout(pad=1.5)
plt.savefig('poly_kernel_models_actual_vs_predicted_subplots.tiff', format='tiff', dpi=1200)
plt.show()

# Print metrics
print("\nPolynomial/Kernel Models Performance Metrics:\n")
for model, metrics in performance.items():
    print(f"{model}: RMSE = {metrics['RMSE']}, R² = {metrics['R²']}, Pearson = {metrics['Pearson']}, Spearman = {metrics['Spearman']}")

# Heatmap of Metrics
performance_df = pd.DataFrame.from_dict(performance, orient='index')
plt.figure(figsize=(8, 5))
sns.set(font_scale=0.9, font='sans-serif')
sns.heatmap(performance_df, annot=True, cmap='YlOrBr', fmt='.6f', linewidth=.5, linecolor='#D3D3D3', cbar_kws={"shrink": 0.8})
plt.tight_layout()
plt.savefig('poly_kernel_models_performance_heatmap.tiff', format='tiff', dpi=1200)
plt.show()

