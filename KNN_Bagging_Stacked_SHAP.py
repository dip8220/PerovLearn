# === Import Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns
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

# === 1. Load Dataset ===
df = pd.read_csv('D:/Final Year Project/Independent_expanded_with_Actual_Bandgap.csv')
X = df[['MA', 'FA', 'Cs', 'Cl', 'Br', 'I', 'Pb', 'Sn']].values
y = df['Bandgap'].values
feature_names = ['MA', 'FA', 'Cs', 'Cl', 'Br', 'I', 'Pb', 'Sn']

# === 2. Train-Test Split and Scaling ===
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

# === 5. Train Stacked Meta-Learners ===
# Base predictions for stacking
def get_preds(model, X, model_type='sk'):
    pred = model.predict(X)
    return pred.ravel() if model_type == 'dl' else pred

# Create base predictions
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

# Train meta-learners
stack_lr = LinearRegression().fit(base_preds_train, y_train_scaled)
stack_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(base_preds_train, y_train_scaled)

# === 6. SHAP Explainers ===
sampled_X = shap.utils.sample(X_test_scaled, 200, random_state=42)

explainer_dict = {
    'KNN': shap.Explainer(knn.predict, X_train_scaled, algorithm='permutation'),
    'Bagging': shap.Explainer(bag.predict, X_train_scaled, algorithm='permutation'),
    'Stacked-LR': shap.Explainer(lambda x: stack_lr.predict(np.vstack([
        get_preds(ridge, x),
        get_preds(svr, x),
        get_preds(bag, x),
        get_preds(knn, x),
        get_preds(mlp, x, 'dl'),
        get_preds(trans, x, 'dl')
    ]).T), X_train_scaled, algorithm='permutation'),
    'Stacked-RF': shap.Explainer(lambda x: stack_rf.predict(np.vstack([
        get_preds(ridge, x),
        get_preds(svr, x),
        get_preds(bag, x),
        get_preds(knn, x),
        get_preds(mlp, x, 'dl'),
        get_preds(trans, x, 'dl')
    ]).T), X_train_scaled, algorithm='permutation')
}

# === 7. SHAP Summary Subplots ===
model_names = ['KNN', 'Bagging', 'Stacked-LR', 'Stacked-RF']
subplot_labels = ['(a)', '(b)', '(c)', '(d)']

plt.figure(figsize=(15, 9))
for i, name in enumerate(model_names):
    shap_values = explainer_dict[name](sampled_X)
    plt.subplot(2, 2, i + 1)
    shap.summary_plot(
        shap_values,
        sampled_X,
        feature_names=feature_names,
        plot_type="dot",
        color=plt.get_cmap("YlOrBr"),
        show=False
    )
    plt.xlabel("SHAP value", fontsize=12, fontfamily='sans-serif')
    plt.xticks(fontsize=10, fontfamily='sans-serif')
    plt.yticks(fontsize=10, fontfamily='sans-serif')
    plt.text(0.5, -0.3, subplot_labels[i], transform=plt.gca().transAxes,
             fontsize=10, ha='center', va='top', fontfamily='sans-serif')
    
plt.tight_layout()
plt.savefig("SHAP_KNN_Bagging_StackedLR_RF.tiff", format='tiff', dpi=1200)
plt.show()
