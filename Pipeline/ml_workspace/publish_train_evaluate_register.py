# ﻿
import argparse
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core import Environment, Dataset
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

from ml_workspace.utils.datastore import get_datastore
from ml_workspace.utils.environment_variables import EnvironmentVariables
from ml_workspace.utils.workspace import get_workspace


def get_or_create_compute(workspace, cpu_cluster_name, compute_vm_size, max_nodes):
    try:
        cpu_cluster = ComputeTarget(workspace=workspace, name=cpu_cluster_name)
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(vm_size=compute_vm_size,
                                                               min_nodes=0,
                                                               max_nodes=max_nodes)
        cpu_cluster = ComputeTarget.create(workspace, cpu_cluster_name, compute_config)

    cpu_cluster.wait_for_completion(show_output=True)
    return cpu_cluster


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_name", type=str)
    return parser.parse_args()


def main():
    env = EnvironmentVariables()
    args = add_arguments()

    workspace = get_workspace()

    cpu_cluster_name = env.cpu_cluster_name
    compute = get_or_create_compute(workspace, cpu_cluster_name, env.compute_vm_size, env.max_nodes)

    environment = Environment.load_from_directory(env.sources_directory_train)
    environment.register(workspace)
    run_configuration = RunConfiguration()
    run_configuration.environment = environment

    model_name_param = PipelineParameter(name="model_name", default_value=env.model_name)
    build_id_param = PipelineParameter(name="build_id", default_value=env.build_id)
    should_tune_hyperparameters_param = PipelineParameter(name="should_tune_hyperparameters",
                                                          default_value=env.should_tune_hyperparameters)
    parallelism_level_param = PipelineParameter(name="parallelism_level", default_value=env.parallelism_level)
    force_register_param = PipelineParameter(name="force_register", default_value=env.force_register)

    datastore = get_datastore()

    dataset_name = env.dataset_name
    dataset_path = env.dataset_path
    print(f"Creating new dataset version for {dataset_name} in datastore {datastore} from file {dataset_path}")
    temp_dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, dataset_path)])
    dataset = temp_dataset.register(
        workspace=workspace,
        name=dataset_name,
        description=dataset_name,
        tags={'format': 'CSV'},
        create_new_version=True)

    train_output = PipelineData(
        'train_output',
        output_name='train_output',
        datastore=datastore)

    train_step = PythonScriptStep(
        name="Train model",
        compute_target=compute,
        script_name=env.train_script_name,
        runconfig=run_configuration,
        inputs=[dataset.as_named_input('training')],
        outputs=[train_output],
        arguments=[
            "--build_id", build_id_param,
            "--model_name", model_name_param,
            "--parallelism_level", parallelism_level_param,
            "--should_tune_hyperparameters", should_tune_hyperparameters_param
        ],
        allow_reuse=False
    )

    evaluate_step = PythonScriptStep(
        name="Evaluate model",
        compute_target=compute,
        script_name=env.evaluate_script_name,
        runconfig=run_configuration,
        inputs=[train_output],
        arguments=[
            "--build_id", build_id_param,
            "--model_name", model_name_param,
            "--train_output", train_output,
            "--force_register", force_register_param
        ],
        allow_reuse=False
    )

    register_step = PythonScriptStep(
        name="Register model",
        compute_target=compute,
        script_name=env.register_script_name,
        runconfig=run_configuration,
        inputs=[train_output],
        arguments=[
            "--build_id", build_id_param,
            "--model_name", model_name_param,
            "--train_output", train_output
        ],
        allow_reuse=False
    )

    evaluate_step.run_after(train_step)
    register_step.run_after(evaluate_step)

    steps = [train_step, evaluate_step, register_step]

    train_pipeline = Pipeline(workspace=workspace, steps=steps)
    train_pipeline.validate()
    published_pipeline = train_pipeline.publish(
        name=env.pipeline_name,
        description="Train/Eval/Register if better pipeline",
        version=env.build_id
    )

    output_file_name = args.output_file_name
    if output_file_name:
        with open(output_file_name, "w") as output_file:
            output_file.write(published_pipeline.id)

    print(f"Published pipeline {published_pipeline.name} for build {published_pipeline.version}")


if __name__ == '__main__':
    main()
