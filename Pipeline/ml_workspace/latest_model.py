import argparse

from ml_workspace.utils.environment_variables import EnvironmentVariables
from ml_workspace.utils.workspace import get_workspace
from azureml.core.model import Model


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_name", type=str)
    return parser.parse_args()


def main():
    args = add_arguments()
    env = EnvironmentVariables()
    workspace = get_workspace()
    models = Model.list(workspace, env.model_name, latest=True)
    latest_model = models[0]
    latest_version = latest_model.version

    output_file_name = args.output_file_name
    if output_file_name:
        with open(output_file_name, "w") as output_file:
            print(f"Latest model version: {str(latest_version)}")
            output_file.write(str(latest_version))


if __name__ == '__main__':
    main()
