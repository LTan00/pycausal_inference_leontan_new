
import pandas as pd
import numpy as np
from patsy import dmatrix
from sklearn.linear_model import LogisticRegression, LinearRegression

def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:
    """
    Estimate the Average Treatment Effect (ATE) using Inverse Propensity Weighting (IPW)
    """
    # Create design matrix for propensity score model
    X = dmatrix(ps_formula, df, return_type='dataframe')
    
    # Fit logistic regression to estimate propensity scores
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X, df[T])
    ps = model.predict_proba(X)[:, 1]
    
    # Clip propensity scores to avoid division by zero
    ps = np.clip(ps, 1e-3, 1-1e-3)
    
    # IPW formula
    T_array = df[T].values
    Y_array = df[Y].values
    ate = np.mean((T_array - ps) / (ps * (1 - ps)) * Y_array)
    
    return ate


def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str) -> float:
    """
    Estimate the Average Treatment Effect (ATE) using Doubly Robust Estimation
    """
    # Design matrix for both propensity score and outcome models
    X = dmatrix(formula, df, return_type='dataframe')
    
    # Propensity score model
    ps_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    ps_model.fit(X, df[T])
    e = ps_model.predict_proba(X)[:, 1]
    e = np.clip(e, 1e-3, 1-1e-3)  # clip for numerical stability
    
    # Outcome models
    Y_array = df[Y].values
    T_array = df[T].values
    
    # Outcome model for treated
    model_treated = LinearRegression()
    model_treated.fit(X[T_array == 1], Y_array[T_array == 1])
    mu1 = model_treated.predict(X)
    
    # Outcome model for control
    model_control = LinearRegression()
    model_control.fit(X[T_array == 0], Y_array[T_array == 0])
    mu0 = model_control.predict(X)
    
    # Doubly robust ATE formula
    dr_ate = np.mean(
        mu1 - mu0 + T_array * (Y_array - mu1) / e - (1 - T_array) * (Y_array - mu0) / (1 - e)
    )
    
    return dr_ate

