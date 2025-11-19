import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

def s_learner_discrete(train: pd.DataFrame, test: pd.DataFrame, X: list, T: str, y: str) -> pd.DataFrame:
    test_out = test.copy()
    X_train = train[X].copy()
    X_train = X_train.assign(**{T: train[T].values})
    y_train = train[y].values
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    X_test_base = test[X].copy()
    X_test_t1 = X_test_base.assign(**{T: np.ones(len(X_test_base), dtype=int)})
    X_test_t0 = X_test_base.assign(**{T: np.zeros(len(X_test_base), dtype=int)})
    mu1 = model.predict(X_test_t1)
    mu0 = model.predict(X_test_t0)
    test_out["cate"] = pd.to_numeric(mu1 - mu0, errors="coerce")
    return test_out

def t_learner_discrete(train: pd.DataFrame, test: pd.DataFrame, X: list, T: str, y: str) -> pd.DataFrame:
    test_out = test.copy()
    control = train[train[T] == 0]
    treated = train[train[T] == 1]
    mu0_model = LGBMRegressor(random_state=42)
    mu1_model = LGBMRegressor(random_state=42)
    if len(control) > 0:
        mu0_model.fit(control[X], control[y])
    else:
        mu0_model.fit(train[X], np.full(len(train), train[y].mean()))
    if len(treated) > 0:
        mu1_model.fit(treated[X], treated[y])
    else:
        mu1_model.fit(train[X], np.full(len(train), train[y].mean()))
    mu0 = mu0_model.predict(test_out[X])
    mu1 = mu1_model.predict(test_out[X])
    test_out["cate"] = pd.to_numeric(mu1 - mu0, errors="coerce")
    return test_out

def x_learner_discrete(train: pd.DataFrame, test: pd.DataFrame, X: list, T: str, y: str) -> pd.DataFrame:
    test_out = test.copy()
    control = train[train[T] == 0]
    treated = train[train[T] == 1]
    mu0_model = LGBMRegressor(random_state=42)
    mu1_model = LGBMRegressor(random_state=42)
    if len(control) > 0:
        mu0_model.fit(control[X], control[y])
    else:
        mu0_model.fit(train[X], np.full(len(train), train[y].mean()))
    if len(treated) > 0:
        mu1_model.fit(treated[X], treated[y])
    else:
        mu1_model.fit(train[X], np.full(len(train), train[y].mean()))
    mu1_on_train = mu1_model.predict(train[X])
    mu0_on_train = mu0_model.predict(train[X])
    tau0 = mu1_on_train - train[y].values
    tau1 = train[y].values - mu0_on_train
    tau0_model = LGBMRegressor(random_state=42)
    tau1_model = LGBMRegressor(random_state=42)
    if len(control) > 0:
        tau0_model.fit(control[X], tau0[train[T] == 0])
    else:
        tau0_model.fit(train[X], np.zeros(len(train)))
    if len(treated) > 0:
        tau1_model.fit(treated[X], tau1[train[T] == 1])
    else:
        tau1_model.fit(train[X], np.zeros(len(train)))
    prop_model = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=10000)
    prop_model.fit(train[X], train[T])
    e_test = np.clip(prop_model.predict_proba(test_out[X])[:, 1], 0.01, 0.99)
    tau0_hat_test = tau0_model.predict(test_out[X])
    tau1_hat_test = tau1_model.predict(test_out[X])
    cate = (1 - e_test) * tau1_hat_test + e_test * tau0_hat_test
    test_out["cate"] = pd.to_numeric(cate, errors="coerce")
    return test_out

def double_ml_cate(train: pd.DataFrame, test: pd.DataFrame, X: list, T: str, y: str) -> pd.DataFrame:
    test_out = test.copy()
    X_train = train[X].reset_index(drop=True)
    T_train = train[T].reset_index(drop=True)
    Y_train = train[y].reset_index(drop=True)
    n = len(train)
    t_hat = np.zeros(n)
    y_hat = np.zeros(n)
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_train):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        t_model = LGBMRegressor(random_state=42)
        y_model = LGBMRegressor(random_state=42)
        t_model.fit(X_tr, T_train.iloc[train_idx])
        y_model.fit(X_tr, Y_train.iloc[train_idx])
        t_hat[val_idx] = t_model.predict(X_val)
        y_hat[val_idx] = y_model.predict(X_val)
    T_res = T_train.values - t_hat
    Y_res = Y_train.values - y_hat
    eps = 1e-6
    T_res_safe = np.where(np.abs(T_res) < eps, np.sign(T_res) * eps + eps, T_res)
    Y_star = Y_res / T_res_safe
    weights = T_res_safe ** 2
    tau_model = LGBMRegressor(random_state=42)
    tau_model.fit(X_train, Y_star, sample_weight=weights)
    tau_test = tau_model.predict(test_out[X])
    test_out["cate"] = pd.to_numeric(tau_test, errors="coerce")
    return test_out


