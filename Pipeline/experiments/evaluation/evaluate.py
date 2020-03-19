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
    parser.add_argument("--force_register", type=str, default="false")
    return parser.parse_args()


def mark_model_eligible_for_registration(run_context):
    run_context.parent.tag("should_register_model", "True")


def deny_new_model_registration(run_context):
    run_context.parent.tag("should_register_model", "False")


def main():
    print("Evaluating model")
    args = add_arguments()

    run_context = Run.get_context()
    model_name = args.model_name

    workspace = run_context.experiment.workspace

    latest_models = Model.list(workspace=workspace, name=model_name, latest=True)
    force_register = args.force_register

    if force_register.lower() == "true":
        print("Forcing new model registration")
        mark_model_eligible_for_registration(run_context)
        sys.exit(0)

    if len(latest_models) == 0:
        print("No previous models found, registering new one")
        mark_model_eligible_for_registration(run_context)
        sys.exit(0)

    latest_model = latest_models[0]
    latest_model_tags = latest_model.tags
    latest_model_mse = float(latest_model_tags["MSE"])
    latest_model_accuracy = float(latest_model_tags["Accuracy"])

    model_container = args.train_output
    model_file_path = os.path.join(model_container, model_name)
    model = joblib.load(model_file_path)

    if model is None:
        print(f"Model {model_name} does not exist")
        sys.exit(0)

    parent_run_tags = run_context.parent.get_tags()
    new_accuracy = float(parent_run_tags["accuracy"])
    new_mse = float(parent_run_tags["mse"])

    print(f"Evaluating model {model_name} with accuracy: {new_accuracy} and mse: {new_mse}")

    if new_mse > latest_model_mse and new_accuracy > latest_model_accuracy:
        mark_model_eligible_for_registration(run_context)
    else:
        deny_new_model_registration(run_context)


if __name__ == '__main__':
    main()
