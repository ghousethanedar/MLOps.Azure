import os
import csv
import pandas as pd
import numpy
from inference_schema.schema_decorators import input_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

from azure.storage.blob import BlockBlobService
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


def to_float(float_candidate):
    replace_comma_to_dot = lambda x: (x.replace(",", "."))
    replaced_value = replace_comma_to_dot(float_candidate)
    parsed = float(replaced_value)
    return parsed


def load_dataset_and_init_scaler():
    account_name = os.environ['STORAGE_ACCOUNT_NAME']
    account_key = os.environ['STORAGE_ACCOUNT_KEY']
    container_name = os.environ['STORAGE_ACCOUNT_CONTAINER']
    file_name = os.environ['STORAGE_ACCOUNT_FILE']
    blob_service_client = BlockBlobService(account_name=account_name, account_key=account_key)

    local_path = os.path.expanduser("~/TempBlob")
    if not os.path.exists(local_path):
        os.makedirs(os.path.expanduser("~/TempBlob"))

    full_path_to_file = os.path.join(local_path, "latest.csv")

    blob_service_client.get_blob_to_path(
        container_name, file_name, full_path_to_file)

    column_names = []
    with open(full_path_to_file, "rt") as f:
        reader = csv.reader(f)
        i = next(reader)
        column_names.extend(i)

    feature_vectors_no_date = column_names.copy()
    del feature_vectors_no_date[0]

    dtypes = {}
    for dtype in feature_vectors_no_date:
        dtypes[dtype] = to_float

    train = pd.read_csv(full_path_to_file
                        , parse_dates=["date"]
                        , converters=dtypes)

    TO_DROP = ['% Iron Concentrate', '% Silica Feed', '% Iron Feed', '% Silica Concentrate', 'date']
    train.drop(TO_DROP, axis=1, inplace=True)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(train)
    del train
    return scaler


def init():
    global model
    global scaler
    scaler = load_dataset_and_init_scaler()
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), "mining_random_forest")
    print(f"Loading model from path {model_path}")
    model = joblib.load(model_path)


input_sample = numpy.array(
    [[3749.88, 534.217, 387.825, 9.6091, 1.68839, 301.257, 303.609, 299.703, 297.7195764576, 297.7805144639, 337.841,
      324.498, 596.128, 608.905, 355.065, 366.537, 335.231, 603.246, 323.066]], dtype=numpy.float32)


@input_schema('data', NumpyParameterType(input_sample))
def run(data):
    print(f"Handling request {data}")
    try:
        x = scaler.transform(data)
        result = model.predict(x)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error


if __name__ == "__main__":
    init()
    print(run(input_sample))
