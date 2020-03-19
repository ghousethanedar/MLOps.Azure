import os
from dotenv import load_dotenv


class Singleton(type):
    instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.instances:
            cls.instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.instances[cls]


class EnvironmentVariables(metaclass=Singleton):
    def __init__(self):
        load_dotenv()
        self._build_id = os.environ.get("BUILD_BUILDID")
        self._subscription_id = os.environ.get("SUBSCRIPTION_ID")
        self._storage_account_key = os.environ.get("STORAGE_ACCOUNT_KEY")
        self._storage_account_name = os.environ.get("STORAGE_ACCOUNT_NAME")
        self._storage_container_name = os.environ.get("STORAGE_CONTAINER_NAME")
        self._workspace_name = os.environ.get("WORKSPACE_NAME")
        self._resource_group = os.environ.get("RESOURCE_GROUP")
        self._train_script_name = os.environ.get("TRAIN_SCRIPT_NAME")
        self._evaluate_script_name = os.environ.get("EVALUATE_SCRIPT_NAME")
        self._dataset_name = os.environ.get("DATASET_NAME")
        self._dataset_path = os.environ.get("DATASET_PATH")
        self._model_name = os.environ.get("MODEL_NAME")
        self._max_nodes = os.environ.get("MAX_NODES")
        self._compute_vm_size = os.environ.get("COMPUTE_VM_SIZE")
        self._cpu_cluster_name = os.environ.get("CPU_CLUSTER_NAME")
        self._pipeline_name = os.environ.get("PIPELINE_NAME")
        self._datastore_name = os.environ.get("DATASTORE_NAME")
        self._experiment_name = os.environ.get("EXPERIMENT_NAME")
        self._register_script_name = os.environ.get("REGISTER_SCRIPT_NAME")
        self._should_tune_hyperparameters = os.environ.get("SHOULD_TUNE_HYPERPARAMETERS")
        self._parallelism_level = os.environ.get("PARALLELISM_LEVEL")
        self._sources_directory_train = os.environ.get("SOURCES_DIRECTORY_TRAIN")
        self._force_register = os.environ.get("FORCE_REGISTER")

    @property
    def workspace_name(self):
        return self._workspace_name

    @property
    def resource_group(self):
        return self._resource_group

    @property
    def subscription_id(self):
        return self._subscription_id

    @property
    def cpu_cluster_name(self):
        return self._cpu_cluster_name

    @property
    def compute_vm_size(self):
        return self._compute_vm_size

    @property
    def max_nodes(self):
        return self._max_nodes

    @property
    def model_name(self):
        return self._model_name

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def dataset_path(self):
        return self._dataset_path

    @property
    def build_id(self):
        return self._build_id

    @property
    def train_script_name(self):
        return self._train_script_name

    @property
    def evaluate_script_name(self):
        return self._evaluate_script_name

    @property
    def pipeline_name(self):
        return self._pipeline_name

    @property
    def datastore_name(self):
        return self._datastore_name

    @property
    def storage_account_name(self):
        return self._storage_account_name

    @property
    def storage_container_name(self):
        return self._storage_container_name

    @property
    def storage_account_key(self):
        return self._storage_account_key

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def register_script_name(self):
        return self._register_script_name

    @property
    def should_tune_hyperparameters(self):
        return self._should_tune_hyperparameters or False

    @property
    def parallelism_level(self):
        return self._parallelism_level or 1

    @property
    def sources_directory_train(self):
        return self._sources_directory_train

    @property
    def force_register(self):
        return self._force_register or "false"
