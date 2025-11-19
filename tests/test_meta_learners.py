import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest
from pycausal_inference.meta_learners import s_learner_discrete, t_learner_discrete, x_learner_discrete, double_ml_cate

def simple_data():
    np.random.seed(42)
    n = 1000
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    prob_t = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    t = np.random.binomial(1, prob_t, n)
    y = 2.0 * t + x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y})
    train = df.iloc[:800].copy()
    test = df.iloc[800:].copy()
    return train, test

def continuous_treatment_data():
    np.random.seed(789)
    n = 1000
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    t = 10 + x1 + 2*x2 + np.random.normal(0, 1, n)
    y = t + x1 + 0.5*x2 + np.random.normal(0, 0.5, n)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y})
    train = df.iloc[:800].copy()
    test = df.iloc[800:].copy()
    return train, test

def test_s_learner_returns_dataframe():
    train, test = simple_data()
    result = s_learner_discrete(train, test, ['x1', 'x2'], 't', 'y')
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(test)

def test_s_learner_has_cate_column():
    train, test = simple_data()
    result = s_learner_discrete(train, test, ['x1', 'x2'], 't', 'y')
    assert 'cate' in result.columns

def test_s_learner_constant_effect():
    train, test = simple_data()
    result = s_learner_discrete(train, test, ['x1', 'x2'], 't', 'y')
    estimated_ate = result['cate'].mean()
    true_effect = 2.0
    tolerance = 0.3
    assert abs(estimated_ate - true_effect) < tolerance

def test_s_learner_return_numeric_cate():
    train, test = simple_data()
    result = s_learner_discrete(train, test, ['x1', 'x2'], 't', 'y')
    assert ptypes.is_numeric_dtype(result['cate'])

def test_s_learner_no_nan_values():
    train, test = simple_data()
    result = s_learner_discrete(train, test, ['x1', 'x2'], 't', 'y')
    assert not result['cate'].isna().any()

def test_t_learner_returns_dataframe():
    train, test = simple_data()
    result = t_learner_discrete(train, test, ['x1', 'x2'], 't', 'y')
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(test)
    assert 'cate' in result.columns
    assert ptypes.is_numeric_dtype(result['cate'])
    assert not result['cate'].isna().any()

def test_x_learner_returns_dataframe():
    train, test = simple_data()
    result = x_learner_discrete(train, test, ['x1', 'x2'], 't', 'y')
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(test)
    assert 'cate' in result.columns
    assert ptypes.is_numeric_dtype(result['cate'])
    assert not result['cate'].isna().any()

def test_double_ml_returns_dataframe():
    train, test = simple_data()
    result = double_ml_cate(train, test, ['x1', 'x2'], 't', 'y')
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(test)
    assert 'cate' in result.columns
    assert ptypes.is_numeric_dtype(result['cate'])
    assert not result['cate'].isna().any()

def test_double_ml_continuous_treatment():
    train, test = continuous_treatment_data()
    result = double_ml_cate(train, test, ['x1', 'x2'], 't', 'y')
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(test)
    assert 'cate' in result.columns
    assert ptypes.is_numeric_dtype(result['cate'])
    assert not result['cate'].isna().any()
    estimated_ate = result['cate'].mean()
    true_effect = 1.0
    tolerance = 0.3
    assert abs(estimated_ate - true_effect) < tolerance
