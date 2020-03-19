from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication

from ml_workspace.utils.environment_variables import EnvironmentVariables


def get_workspace():
    env = EnvironmentVariables()
    cli_auth = AzureCliAuthentication()
    workspace = Workspace(workspace_name=env.workspace_name,
                          subscription_id=env.subscription_id,
                          resource_group=env.resource_group,
                          auth=cli_auth
                          )
    return workspace
