import re
import numpy as np
import pandas as pd

# 1. math functions
def func_pw2(x, A, k):
    # y = A * x ^ k
    return A * np.power(x, k)

def func_pw3(x, A, k, b):
    # y = A * x ^ k + b
    return A * np.power(x, k) + b

def func_exp(x, A, k):
    # y = A ^ x + k
    return A ** x + k

def func_linear(x, k, b):
    return k * x + b

def func_quad2(x, k, b):
    return k * x**2 + b

def func_quad3(x, a, b, c):
    return a * ((x - b) ** 2) + c

def func_tmp(x, m, b, c):
    return m * np.sqrt(1 + np.power(b * x, 2) ) + c


# some utils
def rm_outliers(df, cols, threshold=1.5, lower_only=True):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - (threshold * IQR)
        upper_bound = Q3 + (threshold * IQR)

        if lower_only:
            df = df[df[col] >= lower_bound]
        else:
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df


def extract_floats(s, pattern = r"[-+]?\d*\.\d+e?[-+]?\d*"):
    matches = re.findall(pattern, s)
    floats = [float(num) for num in matches]
    return pd.Series(floats)
