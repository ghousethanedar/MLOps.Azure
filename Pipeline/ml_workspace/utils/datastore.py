from ml_workspace.utils.environment_variables import EnvironmentVariables
from azureml.core import Datastore
from msrest.exceptions import HttpOperationError

from ml_workspace.utils.workspace import get_workspace


def get_datastore():
    env = EnvironmentVariables()
    datastore_name = env.datastore_name
    storage_account_name = env.storage_account_name
    storage_container_name = env.storage_container_name
    storage_account_key = env.storage_account_key
    workspace = get_workspace()

    try:
        datastore = Datastore.get(workspace=workspace, datastore_name=datastore_name)
    except HttpOperationError:
        datastore = Datastore.register_azure_blob_container(
            workspace=workspace,
            datastore_name=datastore_name,
            account_name=storage_account_name,
            container_name=storage_container_name,
            account_key=storage_account_key)

    return datastore
