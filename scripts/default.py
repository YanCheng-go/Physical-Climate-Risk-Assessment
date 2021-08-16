"""
Metadata retrieval or universal functions of EBRD projection.
"""

from sklearn import linear_model


# Metadata retrieval
def switch_X_range(x):
    return {
        'air temperature': (230, 315),
        'precipitation': (0, 0.127),  # in meter
    }.get(x, 'Invalid input!')


# Modelling functions
def switch_models(x, **kwargs):
    if x == 'ols':
        return linear_model.LinearRegression(**kwargs)
    elif x == 'lasso':
        return linear_model.Lasso(alpha=0.55, **kwargs)


# Feature conversion or engineering functions
def pr_to_m(x):
    return x * 86.4


def pr_from_m(x):
    return x / 86.4


def harm_pre_unit(x, input_label):
    return {
        'gddp': pr_to_m,
        'era5': pr_from_m
    }.get(input_label)(x)
