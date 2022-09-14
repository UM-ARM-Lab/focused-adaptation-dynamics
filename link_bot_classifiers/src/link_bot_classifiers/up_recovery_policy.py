import pathlib
from typing import Dict

import numpy as np

from link_bot_classifiers.base_recovery_policy import BaseRecoveryPolicy
from link_bot_pycommon.experiment_scenario import ExperimentScenario


class UpRecoveryPolicy(BaseRecoveryPolicy):

    def __init__(self, path: pathlib.Path, scenario: ExperimentScenario, rng: np.random.RandomState, u: Dict):
        super().__init__(path, scenario, rng, u)

    def __call__(self, environment: Dict, state: Dict):
        action, _ = self.scenario.sample_action(action_rng=self.rng, environment=environment, state=state,
                                                action_params=self.params, validate=True)
        for i in range(100):
            left_dz = action['left_gripper_delta_position'][2]
            right_dz = action['right_gripper_delta_position'][2]
            if left_dz > 0.02 and right_dz > 0.02:
                break

            action, _ = self.scenario.sample_action(action_rng=self.rng, environment=environment, state=state,
                                                    action_params=self.params, validate=True)
        return action
