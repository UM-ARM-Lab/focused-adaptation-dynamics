import pathlib
from functools import lru_cache
from typing import Optional

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from state_space_dynamics.torch_udnn_dynamics_wrapper import TorchUDNNDynamicsWrapper


@lru_cache
def load_generic_model(model_dir: pathlib.Path, scenario: Optional[ExperimentScenario] = None):
    checkpoint = model_dir.as_posix()[2:]
    model = TorchUDNNDynamicsWrapper(checkpoint, scenario)
    return model
