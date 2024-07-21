import os
import logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor

from sklearn import preprocessing
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

data_dir = f"/home/zhaoy/asset-fastCAE/dataset/vvenc"
rlt_dir  = "/home/zhaoy/asset-fastCAE/results/vvenc/tables/predCurve"
os.makedirs(rlt_dir, exist_ok=True)


def build_keras_regressor(input_dim, output_dim, optimizer="adam"):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim))

    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer)
    return model


def _single_regression(train_df, test_df, preset_x, regressor_cls, regressor_params, metrics=None, scale=True, verbose=False):
    """
    Train and Evaluate -> ML-based curve param regression
    """
    metrics = ["p1", "p2"] if metrics is None else metrics
    pred_metrics = [f"pred_{x}" for x in metrics]

    train_df = train_df.sort_values(by=["preset", "seqName", "sceneId", "size"]).reset_index(drop=True)
    test_df = test_df.sort_values(by=["preset", "seqName", "sceneId", "size"]).reset_index(drop=True)

    train_x = train_df[train_df["preset"] == preset_x][metrics + [f"y{i}" for i in range(5)]].astype(float)
    test_x  = test_df[test_df["preset"] == preset_x][metrics + [f"y{i}" for i in range(5)]].astype(float)
    assert (train_x.empty == False) and (test_x.empty == False)

    regressor_name = regressor_cls if regressor_cls in ["Adam", "RMSProp"] else regressor_cls.__name__

    rows  = []
    pred_dfs = []
    for preset in train_df["preset"].unique():
        train_y = train_df[(train_df["preset"] == preset)][metrics].astype(float)
        test_y  = test_df[(test_df["preset"] == preset)][metrics].astype(float)

        if train_y.empty or test_y.empty:
            continue

        pred_df = test_df[(test_df["preset"] == preset)][["seqName", "sceneId", "size"]]
        pred_df["regressor"] = regressor_name
        pred_df["input"]  = preset_x
        pred_df["preset"] = preset
        pred_df[metrics]  = test_y

        if preset == preset_x:
            pred_df[pred_metrics] = test_y
            pred_dfs.append(pred_df)
            continue

        # scaling data
        if scale:
            scaler_x = preprocessing.MinMaxScaler()
            train_x  = scaler_x.fit_transform(train_x)
            test_x   = scaler_x.fit_transform(test_x)

            scaler_y = preprocessing.MinMaxScaler()
            train_y  = scaler_y.fit_transform(train_y)
            test_y   = scaler_y.fit_transform(test_y)

        # regression
        if regressor_cls in ["Adam", "RMSProp"]:
            optimizer = Adam() if regressor_cls == "Adam" else RMSprop()
            regressor = build_keras_regressor(input_dim=len(metrics) + 5, output_dim=len(metrics), optimizer=optimizer)
            regressor.fit(train_x, train_y, **regressor_params)
        else:
            regressor = MultiOutputRegressor(regressor_cls(**regressor_params))
            regressor.fit(train_x, train_y)

        # training performance
        train_y_pred = regressor.predict(train_x)
        if scale:
            train_y_pred = scaler_y.inverse_transform(train_y_pred)
            train_y = scaler_y.inverse_transform(train_y)

        train_r2  = r2_score(train_y, train_y_pred, multioutput="raw_values")
        train_rmse = mean_squared_error(train_y, train_y_pred, squared=False) / len(train_y)
        train_evs = explained_variance_score(train_y, train_y_pred, multioutput="raw_values")

        # testing performance
        test_y_pred = regressor.predict(test_x)
        if scale:
            test_y_pred = scaler_y.inverse_transform(test_y_pred)
            test_y = scaler_y.inverse_transform(test_y)
        pred_df[pred_metrics] = test_y_pred
        pred_dfs.append(pred_df)

        test_r2 = r2_score(test_y, test_y_pred, multioutput="raw_values")
        test_rmse = mean_squared_error(test_y, test_y_pred, squared=False) / len(test_y)
        test_evs = explained_variance_score(test_y, test_y_pred, multioutput="raw_values")

        if verbose:
            print(f"preset: {preset}, input: {preset_x}, regressor: {regressor_name}, train_rmse: {train_rmse}, train_r2: {train_r2}, train_evs: {train_evs}, test_rmse: {test_rmse}, test_r2: {test_r2}, test_evs: {test_evs}")

        rows.append([preset, f"{preset_x}", regressor_name, train_rmse, train_r2, train_evs, test_rmse, test_r2, test_evs])

    rlts  = pd.DataFrame(rows, columns=["preset", "input", "regressor", "train_rmse", "train_r2", "train_evs", "test_rmse", "test_r2", "test_evs"])
    preds = pd.concat(pred_dfs, axis=0).reset_index(drop=True)

    return rlts, preds


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
regressors = {
    RandomForestRegressor: {'n_estimators': 1000, 'max_depth': 8, 'min_samples_split': 2, 'min_samples_leaf': 1,
                            'random_state': 42},
    LinearRegression: {},
    LinearSVR: {'C': 1.0, 'epsilon': 0.1, 'max_iter': 30000, 'tol': 1e-4, 'random_state': 42},
    SGDRegressor: {'loss': 'huber', 'penalty': 'l1', 'alpha': 0.001, 'learning_rate': 'optimal', 'max_iter': 30000,
                   'tol': 1e-4, 'random_state': 42},
    'Adam': {'epochs': 5000, 'batch_size': 128, 'verbose': False, 'validation_split': 0.1,
             'callbacks': [early_stopping]},
    'RMSProp': {'epochs': 5000, 'batch_size': 128, 'verbose': False, 'validation_split': 0.1,
                'callbacks': [early_stopping]}
}


def _multi_regression(
        target,
        func="quadratic3",
        preset_x="faster",
):
    train_df = pd.read_csv(f"{data_dir}/corr_{func}/corr_{target}_train.csv")
    test_df = pd.read_csv(f"{data_dir}/corr_{func}/corr_{target}_test.csv")

    metrics = ["p1", "p2", "p3"] if func in ["quadratic3"] else ["p1", "p2"]

    all_rlts = []
    all_preds = []
    for regressor_cls, regressor_params in regressors.items():
        try:
            rlts, preds = _single_regression(
                train_df, test_df,
                preset_x=preset_x,
                regressor_cls=regressor_cls,
                regressor_params=regressor_params,
                metrics=metrics,
                scale=True,
                verbose=True
            )
        except Exception as e:
            print(f"{e} ({target}, {func}, {preset_x}, {regressor_cls})")
            continue

        all_rlts.append(rlts)
        all_preds.append(preds)

    rlts_df = pd.concat(all_rlts, axis=0)
    rlts_df["func"] = func
    rlts_df["target"] = target

    preds_df = pd.concat(all_preds, axis=0)
    preds_df["func"] = func
    preds_df["target"] = target

    os.makedirs(f"{rlt_dir}/corr_{func}", exist_ok=True)
    rlts_df.to_csv(f"{rlt_dir}/corr_{func}/rlt_{target}.csv", index=False)
    preds_df.to_csv(f"{rlt_dir}/corr_{func}/pred_{target}.csv", index=False)

    return rlts_df, preds_df


def process_task(params):
    target, func, preset_x = params
    rlts_df, preds_df = _multi_regression(target=target, func=func, preset_x=preset_x)
    return rlts_df, preds_df


def DEBUG():
    for target in ["bitrate", "log2bitrate", "psnr", "log2psnr", "ssim", "log2ssim", "vmaf", "log2vmaf"]:
        for func in ["linear", "power", "quadratic2"]:
            for preset_x in ["faster", "medium", "slower"]:
                process_task((target, func, preset_x))


def PARALLEL_RUN():
    tasks = []
    for target in ["bitrate", "log2bitrate", "psnr", "log2psnr", "ssim", "log2ssim", "vmaf", "log2vmaf"]:
        for func in ["linear", "power", "quadratic2"]:
            for preset_x in tqdm(["faster", "medium", "slower"]):
                tasks.append((target, func, preset_x))

    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_task, tasks)

    all_rlts = [rlt[0] for rlt in results]
    all_preds = [rlt[1] for rlt in results]

    all_rlts_df = pd.concat(all_rlts, axis=0).reset_index(drop=True)
    all_preds_df = pd.concat(all_preds, axis=0).reset_index(drop=True)

    return all_rlts_df, all_preds_df



if __name__ == '__main__':
    all_rlts_df, all_preds_df = PARALLEL_RUN()

    all_rlts_df.to_csv(f"{rlt_dir}/all_rlts.csv", index=False)
    all_preds_df.to_csv(f"{rlt_dir}/all_preds.csv", index=False)