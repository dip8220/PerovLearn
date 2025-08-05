import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras import backend as K

# Load dataset
data = pd.read_csv(r'D:\Final Year Project\Independent_expanded.csv')
X = data[['MA', 'FA', 'Cs', 'Cl', 'Br', 'I', 'Pb', 'Sn']]
y = data['Bandgap']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# RBF Layer definition
class RBFLayer(Layer):
    def __init__(self, units, gamma=None, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = gamma

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.units, input_shape[1]),
                                       initializer='uniform',
                                       trainable=True)
        if self.gamma is None:
            self.gamma = 1.0
        self.gamma = K.variable(self.gamma)

    def call(self, inputs):
        diff = K.expand_dims(inputs, 1) - self.centers
        l2 = K.sum(K.square(diff), axis=-1)
        return K.exp(-self.gamma * l2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

# RBFN model builder
def get_rbfn(units=50, gamma=0.5):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(RBFLayer(units=units, gamma=gamma))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# DL fit/predict
def fit_predict_dl(model):
    model.fit(X_train, y_train_scaled, epochs=100, batch_size=32, verbose=0)
    y_pred_scaled = model.predict(X_test).ravel()
    return y_pred_scaled

# Define models
models = {
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'RBFN': get_rbfn(units=50, gamma=0.5)
}

performance = {}
sample_indices = np.random.choice(len(y_test), size=200, replace=False)
filtered_y_test = y_test.values[sample_indices]

# Subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes = axes.flatten()
subplot_labels = ['(a)', '(b)']

for i, (name, model) in enumerate(models.items()):
    if name == 'RBFN':
        y_pred_scaled = fit_predict_dl(model)
    else:
        model.fit(X_train, y_train_scaled)
        y_pred_scaled = model.predict(X_test)

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

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

    # Scatter plots with consistent style
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
plt.savefig('instance_models_actual_vs_predicted_split.tiff', format='tiff', dpi=1200)
plt.show()

# Print metrics
print("\nInstance-Based Models Performance Metrics:\n")
for model, metrics in performance.items():
    print(f"{model}: RMSE = {metrics['RMSE']}, R² = {metrics['R²']}, Pearson = {metrics['Pearson']}, Spearman = {metrics['Spearman']}")

# Heatmap
performance_df = pd.DataFrame.from_dict(performance, orient='index')
plt.figure(figsize=(8, 5))
sns.heatmap(performance_df, annot=True, cmap='YlOrBr', fmt='.6f', linewidth=.5, linecolor='#D3D3D3', cbar_kws={"shrink": 0.8})
plt.title('Instance-Based Models Performance Heatmap')
plt.tight_layout()
plt.savefig('instance_models_performance_heatmap.tiff', format='tiff', dpi=1200)
plt.show()

