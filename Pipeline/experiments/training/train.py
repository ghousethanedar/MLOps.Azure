from azureml.core.run import Run

import pandas as pd
import numpy as np
import argparse
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

PREDICT_KEY = "% Silica Concentrate"
HIGHLY_CORELATED_VECTORS = ['% Iron Concentrate', '% Silica Feed', '% Iron Feed']


def remove_outliers(df, features, outlier_factor):
    removal_candidates_idx = []

    for col in features:
        series_desc = df.describe()[col]
        iqr = series_desc['75%'] - series_desc['25%']
        lower_bound = series_desc['25%'] - (outlier_factor * iqr)
        upper_bound = series_desc['75%'] + (outlier_factor * iqr)
        removal_candidates_idx.extend(df[df[col] < lower_bound].index.tolist())
        removal_candidates_idx.extend(df[df[col] > upper_bound].index.tolist())

    removal_idx = np.unique(np.array(removal_candidates_idx))
    items_to_remove = len(removal_idx)

    print(
        f"Percent of outliners candidate for IQR ratio {outlier_factor} is {float(items_to_remove / df.shape[0]) * 100}%")

    _ = [df.drop(idx, inplace=True) for idx in removal_idx]
    _ = df.reset_index(inplace=True, drop=True)


def parse_arguments(parser_name):
    parser = argparse.ArgumentParser(parser_name)
    parser.add_argument("--build_id", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--should_tune_hyperparameters", type=str, default="False")
    parser.add_argument("--parallelism_level", type=str, default="1")
    args = parser.parse_args()
    return args


def evaluate(run, model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    mse = mean_squared_error(predictions, test_labels)
    print('Model Performance')
    print('Average Error: {:0.4f}.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('Mean Squared Error = {:0.2f}%.'.format(mse))
    run.log("mse", mse)
    run.log("accuracy", accuracy)
    run.parent.log("mse", mse)
    run.parent.log("accuracy", accuracy)
    result = {"accuracy": accuracy, "mse": mse}
    return result


def train_model_and_tune_hyperparameters(run, data, params_grid, parallelism):
    print(f'hyperparams {params_grid}')
    rfr = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=rfr, param_grid=params_grid,
                               cv=5, n_jobs=parallelism, verbose=1)

    grid_search.fit(data['train']['X'], data['train']['y'])
    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(run, best_grid, data['test']['X'], data['test']['y'])
    grid_accuracy["model"] = best_grid
    return grid_accuracy


def train_model(run, data, params_grid):
    print(f'hyperparams {params_grid}')
    n_estimators = params_grid['n_estimators']
    random_state = params_grid['random_state']
    rfr = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rfr.fit(data['train']['X'], data['train']['y'])
    grid_accuracy = evaluate(run, rfr, data['test']['X'], data['test']['y'])
    grid_accuracy["model"] = rfr
    return grid_accuracy


def prepare_input_for_training(ds):
    x_train = ds.copy().drop([PREDICT_KEY, 'date'], axis=1)
    y_train = ds[PREDICT_KEY].copy()

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train), index=x_train.index, columns=x_train.columns)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=2)

    x_train_2 = x_train.drop(HIGHLY_CORELATED_VECTORS, axis=1)
    x_test_2 = x_test.drop(HIGHLY_CORELATED_VECTORS, axis=1)

    data = {
        "train": {"X": x_train_2, "y": y_train},
        "test": {"X": x_test_2, "y": y_test}
    }
    return data


def normalize_values(ds, features):
    for column in features:
        ds[column] = ds[column].str.replace(",", ".")

    for column in features:
        ds[column] = ds[column].astype('float')


def main():
    args = parse_arguments("train")
    run_context = Run.get_context()

    model_name = args.model_name
    build_id = args.build_id
    should_tune_hyperparameters = args.should_tune_hyperparameters.lower() == 'true'

    dataset = run_context.input_datasets['training']
    train = dataset.to_pandas_dataframe()
    column_names = train.columns
    feature_vectors_no_date = column_names.copy()
    feature_vectors_no_date = feature_vectors_no_date.drop('date')
    normalize_values(train, feature_vectors_no_date)
    remove_outliers(train, feature_vectors_no_date, 2.5)
    train_test_data_set = prepare_input_for_training(train)

    if should_tune_hyperparameters:
        param_grid = {
            'bootstrap': [True],
            'max_depth': [90, 100, 110],
            'max_features': [10, 19, 5],
            'min_samples_leaf': [1, 3, 5],
            'min_samples_split': [2, 4, 10],
            'n_estimators': [100, 200, 300, 1000]
        }
        parallelism_level = int(args.parallelism_level)
        result = train_model_and_tune_hyperparameters(run_context, train_test_data_set, param_grid, parallelism_level)
    else:
        param_grid = {
            'n_estimators': 100,
            'random_state': 2
        }
        result = train_model(run_context, train_test_data_set, param_grid)

    rfr = result["model"]

    output_path_base = os.environ.get("AZUREML_DATAREFERENCE_train_output")
    os.makedirs(output_path_base, exist_ok=True)
    output_path = os.path.join(output_path_base, model_name)
    joblib.dump(value=rfr, filename=output_path)
    print(f"Uploaded the model {model_name} to mounted path {output_path} for experiment {run_context.experiment.name}")
    run_context.parent.tag("accuracy", result["accuracy"])
    run_context.parent.tag("mse", result["mse"])
    run_context.tag("BuildId", value=build_id)
    run_context.tag("run_type", value="train")
    run_context.complete()


if __name__ == '__main__':
    main()
