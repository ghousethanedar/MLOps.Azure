import argparse
import os
import sys

from azureml.core import Run
from azureml.core.model import Model
from sklearn.externals import joblib


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_id", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--train_output", type=str)
    return parser.parse_args()


def main():
    args = add_arguments()

    run_context = Run.get_context()
    experiment = run_context.experiment
    build_id = args.build_id
    model_name = args.model_name
    model_container = args.train_output
    model_file_path = os.path.join(model_container, model_name)
    model = joblib.load(model_file_path)

    metrics = run_context.parent.get_metrics()
    mse = metrics["mse"]
    accuracy = metrics["accuracy"]

    parent_tags = run_context.parent.get_tags()
    try:
        is_new_model_performing_better = parent_tags["should_register_model"].lower() == 'true'
    except KeyError:
        is_new_model_performing_better = True

    if not is_new_model_performing_better:
        print(f"New model performs worse. New mse: {mse}, new accuracy: {accuracy}")
        sys.exit(0)

    if model is None:
        print(f"Model {model_name} not found. Skipping model registration.")
        sys.exit(0)

    tags = {"BuildId": build_id,
            "Accuracy": accuracy,
            "MSE": mse}

    model = Model.register(
        workspace=experiment.workspace,
        model_name=model_name,
        model_path=model_file_path,
        tags=tags)

    print(f"Model {model.name} registered with description {model.description}. Model version is: {model.version}")


if __name__ == "__main__":
    main()
