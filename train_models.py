import os
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


def main():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs('models', exist_ok=True)

    scaler_lr = StandardScaler()
    X_train_lr = scaler_lr.fit_transform(X_train)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_lr, y_train)
    joblib.dump(lr, 'models/logistic_regression_model.joblib')
    joblib.dump(scaler_lr, 'models/logistic_regression_scaler.joblib')

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'models/random_forest_model.joblib')

    lgbm = lgb.LGBMClassifier(random_state=42)
    lgbm.fit(X_train, y_train)
    joblib.dump(lgbm, 'models/lightgbm_model.joblib')

    scaler_km = StandardScaler()
    X_scaled_full = scaler_km.fit_transform(X)

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km.fit(X_scaled_full)
    joblib.dump(km, 'models/kmeans_model.joblib')
    joblib.dump(scaler_km, 'models/kmeans_scaler.joblib')


if __name__ == '__main__':
    main()
