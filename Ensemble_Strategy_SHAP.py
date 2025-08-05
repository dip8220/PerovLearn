# === Import Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# === 1. Load Data ===
df = pd.read_csv('D:/Final Year Project/Independent_expanded.csv')
X = df[['MA', 'FA', 'Cs', 'Cl', 'Br', 'I', 'Pb', 'Sn']].values
y = df['Bandgap'].values
feature_names = ['MA', 'FA', 'Cs', 'Cl', 'Br', 'I', 'Pb', 'Sn']

# === 2. Preprocessing ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler().fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# === 3. Define DL Models ===
def build_mlp():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_transformer():
    inp = Input(shape=(X_train.shape[1],))
    x = Dense(64, activation='relu')(inp)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(), loss='mse')
    return model

# === 4. Train Base Models ===
ridge = Ridge().fit(X_train_scaled, y_train_scaled)
svr = SVR().fit(X_train_scaled, y_train_scaled)
bag = BaggingRegressor(n_estimators=100, random_state=42).fit(X_train_scaled, y_train_scaled)
knn = KNeighborsRegressor().fit(X_train_scaled, y_train_scaled)
mlp = build_mlp(); mlp.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, verbose=0)
trans = build_transformer(); trans.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, verbose=0)

# === 5. Generate Base Model Predictions ===
def get_preds(model, X, model_type='sk'):
    pred = model.predict(X)
    return pred.ravel() if model_type == 'dl' else pred

base_preds_train = np.vstack([
    get_preds(ridge, X_train_scaled),
    get_preds(svr, X_train_scaled),
    get_preds(bag, X_train_scaled),
    get_preds(knn, X_train_scaled),
    get_preds(mlp, X_train_scaled, 'dl'),
    get_preds(trans, X_train_scaled, 'dl')
]).T

base_preds_test = np.vstack([
    get_preds(ridge, X_test_scaled),
    get_preds(svr, X_test_scaled),
    get_preds(bag, X_test_scaled),
    get_preds(knn, X_test_scaled),
    get_preds(mlp, X_test_scaled, 'dl'),
    get_preds(trans, X_test_scaled, 'dl')
]).T

# === 6. Train Meta Models ===
blend_lr_feat = LinearRegression().fit(X_train_scaled, y_train_scaled)  # Direct input-based LR
stack_lr_feat = LinearRegression().fit(base_preds_train, y_train_scaled)  # Prediction-based LR
stack_rf_feat = RandomForestRegressor(n_estimators=100, random_state=42).fit(base_preds_train, y_train_scaled)

# === 7. Evaluate Models ===
def evaluate(y_true, y_pred, name):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:12s} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

evaluate(y_test, scaler_y.inverse_transform(blend_lr_feat.predict(X_test_scaled).reshape(-1, 1)), "Blended-LR")
evaluate(y_test, scaler_y.inverse_transform(stack_lr_feat.predict(base_preds_test).reshape(-1, 1)), "Stacked-LR")
evaluate(y_test, scaler_y.inverse_transform(stack_rf_feat.predict(base_preds_test).reshape(-1, 1)), "Stacked-RF")

# === 8. Setup SHAP Explainers ===
sampled_X = shap.utils.sample(X_test_scaled, 200, random_state=42)

explainer_blend = shap.Explainer(blend_lr_feat.predict, X_train_scaled, algorithm='permutation')

explainer_stack_lr = shap.Explainer(
    lambda x: stack_lr_feat.predict(np.vstack([
        get_preds(ridge, x),
        get_preds(svr, x),
        get_preds(bag, x),
        get_preds(knn, x),
        get_preds(mlp, x, 'dl'),
        get_preds(trans, x, 'dl')
    ]).T),
    X_train_scaled, algorithm='permutation'
)

explainer_stack_rf = shap.Explainer(
    lambda x: stack_rf_feat.predict(np.vstack([
        get_preds(ridge, x),
        get_preds(svr, x),
        get_preds(bag, x),
        get_preds(knn, x),
        get_preds(mlp, x, 'dl'),
        get_preds(trans, x, 'dl')
    ]).T),
    X_train_scaled, algorithm='permutation'
)

# === 9. SHAP Summary Plots ===
shap_values_blend = explainer_blend(sampled_X)
shap_values_stack_lr = explainer_stack_lr(sampled_X)
shap_values_stack_rf = explainer_stack_rf(sampled_X)

shap_values_all = [shap_values_blend, shap_values_stack_lr, shap_values_stack_rf]
titles = ['Blended-LR', 'Stacked-LR', 'Stacked-RF']
subplot_labels = ['(a)', '(b)', '(c)']

plt.figure(figsize=(15, 7))
for i in range(3):
    plt.subplot(2, 2, i + 1)
    shap.summary_plot(
        shap_values_all[i],
        sampled_X,
        feature_names=feature_names,
        plot_type='dot',
        color=plt.get_cmap("YlOrBr"),
        show=False
    )
    plt.xlabel("SHAP value", fontsize=12, fontfamily='sans-serif')
    plt.xticks(fontsize=10, fontfamily='sans-serif')
    plt.yticks(fontsize=10, fontfamily='sans-serif')
    plt.text(0.5, -0.3, subplot_labels[i], transform=plt.gca().transAxes,
             fontsize=10, ha='center', va='top', fontfamily='sans-serif')
    
plt.tight_layout()
plt.savefig("SHAP_Blended_Stacked_LR_RF.tiff", format='tiff', dpi=1200)
plt.show()

