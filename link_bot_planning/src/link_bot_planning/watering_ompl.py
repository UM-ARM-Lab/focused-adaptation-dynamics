import warnings
warnings.simplefilter("once", category=UserWarning)
from typing import Dict

import numpy as np
from link_bot_pycommon.scenario_ompl import ScenarioOmpl
from link_bot_pycommon.water_scenario import WaterSimScenario

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.base as ob
    import ompl.control as oc


class WateringOmpl(ScenarioOmpl):

    def __init__(self, scenario: WaterSimScenario, *args, **kwargs):
        super().__init__(scenario, *args, **kwargs)
        warnings.warn("Warning: target container pos hard coded")
        self._target_container_pos = np.array([0.5, 0])

    @staticmethod
    def numpy_to_ompl_state(state_np: Dict, state_out: ob.CompoundState):
        for i in range(2):
            state_out[0][i] = np.float64(state_np['controlled_container_pos'][i])
        state_out[0][2] = np.float64(state_np['controlled_container_angle'][0])
        for i in range(2):
            state_out[1][i] = np.float64(state_np['target_container_pos'][i])
        state_out[1][2] = np.float64(0)
        state_out[2][0] = np.float64(state_np['control_volume'][0])
        state_out[3][0] = np.float64(state_np['target_volume'][0])
        state_out[4][0] = np.float64(state_np['num_diverged'][0])

    @staticmethod
    def ompl_state_to_numpy(ompl_state: ob.CompoundState):
        ompl_state = {
            'controlled_container_pos': np.array([ompl_state[0][0], ompl_state[0][1]]),
            'target_container_pos': np.array([ompl_state[1][0], ompl_state[1][1]]),
            'controlled_container_angle': np.array([ompl_state[0][2]]),
            'control_volume': np.array([ompl_state[2][0]]),
            'target_volume': np.array([ompl_state[3][0]]),
            'num_diverged': np.array([ompl_state[4][0]]),
            'error': np.zeros(1, dtype=np.float64),
        }
        return ompl_state

    def ompl_control_to_numpy(self, ompl_state: ob.CompoundState, ompl_control: oc.CompoundControl):
        state_np = self.ompl_state_to_numpy(ompl_state)
        current_pos = state_np['controlled_container_pos']
        current_angle = state_np['controlled_container_angle']
        delta_pos = np.array([ompl_control[0][0], ompl_control[0][1]])
        delta_angle = np.array(ompl_control[0][2])
        target_pos = current_pos + delta_pos
        target_angle = current_angle + delta_angle
        assert len(target_pos.shape) == len(target_angle.shape)

        return {
            'controlled_container_target_pos': target_pos,
            'controlled_container_target_angle': target_angle,
        }

    def make_goal_region(self,
                         si: oc.SpaceInformation,
                         rng: np.random.RandomState,
                         params: Dict, goal: Dict,
                         use_torch: bool,
                         plot: bool):
        return WateringGoalRegion(si=si,
                                  scenario_ompl=self,
                                  rng=rng,
                                  threshold=params['goal_params']['threshold'],
                                  goal=goal,
                                  target_container_pos=self._target_container_pos,
                                  plot=plot)

    def print_oob(self, state: Dict):
        print(state)

    def make_state_space(self):
        state_space = ob.CompoundStateSpace()

        self.add_container_subspace(state_space, 'controlled_container')
        self.add_container_subspace(state_space, 'target_container')
        self.add_volume_subspace(state_space, 'control_volume')
        self.add_volume_subspace(state_space, 'target_volume')

        # extra subspace component for the number of diverged steps
        num_diverged_subspace = ob.RealVectorStateSpace(1)
        num_diverged_bounds = ob.RealVectorBounds(1)
        num_diverged_bounds.setLow(-1000)
        num_diverged_bounds.setHigh(1000)
        num_diverged_subspace.setBounds(num_diverged_bounds)
        num_diverged_subspace.setName("num_diverged")
        state_space.addSubspace(num_diverged_subspace, weight=0)

        def _state_sampler_allocator(state_space):
            return WateringStateSampler(state_space,
                                        scenario_ompl=self,
                                        extent=self.planner_params['state_sampler_extent'],
                                        rng=self.state_sampler_rng,
                                        target_container_pos=self._target_container_pos,
                                        plot=self.plot)

        state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(_state_sampler_allocator))

        return state_space

    def add_container_subspace(self, state_space, name="container"):
        min_x, max_x, min_y, max_y, min_z, max_z = self.planner_params['state_extent']
        container_subspace = ob.RealVectorStateSpace(3)
        container_bounds = ob.RealVectorBounds(3)
        # these bounds are not used for sampling
        container_bounds.setLow(0, min_x)
        container_bounds.setHigh(0, max_x)
        container_bounds.setLow(1, min_y)
        container_bounds.setHigh(1, max_y)
        container_bounds.setLow(2, min_z)
        container_bounds.setHigh(2, max_z)
        container_subspace.setBounds(container_bounds)
        container_subspace.setName(name)
        state_space.addSubspace(container_subspace, weight=1)

    def add_volume_subspace(self, state_space, name="container"):
        volume_subspace = ob.RealVectorStateSpace(1)
        volume_bounds = ob.RealVectorBounds(1)
        volume_bounds.setLow(-2)  # more conservative to allow for some dynamics errors
        volume_bounds.setHigh(2)
        volume_subspace.setBounds(volume_bounds)
        volume_subspace.setName(name)
        state_space.addSubspace(volume_subspace, weight=1)

    def make_control_space(self):
        control_space = oc.CompoundControlSpace(self.state_space)

        controlled_container_control_space = oc.RealVectorControlSpace(self.state_space, 3)
        controlled_container_control_bounds = ob.RealVectorBounds(3)
        controlled_container_control_bounds.setLow(0, -0.15)
        controlled_container_control_bounds.setHigh(0, 0.15)
        controlled_container_control_bounds.setLow(1, -0.15)
        controlled_container_control_bounds.setHigh(1, 0.15)
        controlled_container_control_bounds.setHigh(2, -4)
        controlled_container_control_bounds.setHigh(2, 4)
        controlled_container_control_space.setBounds(controlled_container_control_bounds)
        control_space.addSubspace(controlled_container_control_space)

        def _allocator(cs):
            return WateringControlSampler(cs,
                                          scenario_ompl=self,
                                          rng=self.control_sampler_rng,
                                          action_params=self.action_params)

        control_space.setControlSamplerAllocator(oc.ControlSamplerAllocator(_allocator))

        return control_space


class WateringControlSampler(oc.ControlSampler):
    def __init__(self,
                 control_space: oc.CompoundControlSpace,
                 scenario_ompl: WateringOmpl,
                 rng: np.random.RandomState,
                 action_params: Dict):
        super().__init__(control_space)
        self.scenario_ompl = scenario_ompl
        self.rng = rng  # extra subspace component for the number of diverged steps
        self.control_space = control_space
        self.action_params = action_params
        self.max_d_control = self.action_params["max_distance_controlled_container_can_move_per_dim"]

    def sampleNext(self, control_out, previous_control, state):
        is_pour = self.rng.randint(2)
        undo_angle = round(state[0][2], 2)
        state_np = self.scenario_ompl.ompl_state_to_numpy(state)
        can_pour = self.scenario_ompl.s.is_pour_valid_for_state(state_np)
        if is_pour and can_pour:
            control_out[0][0] = 0
            control_out[0][1] = 0
            control_out[0][2] = self.rng.uniform(self.action_params["theta_min"],
                                                 self.action_params["theta_max"]) - undo_angle
        else:
            control_out[0][2] = -undo_angle
            min_action_norm = 0.08
            for i in range(40):
                if state_np["controlled_container_pos"][0] < 0.05:
                    control_out[0][0] = self.rng.uniform(0, self.max_d_control)
                else:
                    control_out[0][0] = self.rng.uniform(-1 * self.max_d_control, self.max_d_control)
                control_out[0][1] = self.rng.uniform(-1.0 * self.max_d_control, self.max_d_control)
                if np.linalg.norm([control_out[0][0], control_out[0][1]]) > min_action_norm:
                    break
            # print(state_np["controlled_container_pos"])

    def sampleStepCount(self, min_steps, max_steps):
        step_count = self.rng.randint(min_steps, max_steps)
        return step_count


class WateringStateSampler(ob.RealVectorStateSampler):

    def __init__(self,
                 state_space,
                 scenario_ompl: WateringOmpl,
                 extent,
                 target_container_pos,
                 rng: np.random.RandomState,
                 plot: bool):
        super().__init__(state_space)
        self._target_container_pos = target_container_pos
        self.scenario_ompl = scenario_ompl
        self.rng = rng
        self.plot = plot
        self.extent = extent

    def sampleUniform(self, state_out: ob.CompoundState):
        min_x, max_x, min_y, max_y, min_theta, max_theta = self.extent
        random_point_x = self.rng.uniform(min_x, max_x)
        random_point_y = self.rng.uniform(min_y, max_y)
        random_angle = self.rng.uniform(min_theta, max_theta)
        random_control_volume = self.rng.uniform(0, 1)
        state_np = {
            'controlled_container_pos': np.array([random_point_x, random_point_y]),
            'controlled_container_angle': np.array([random_angle]),
            'target_container_pos': self._target_container_pos,
            'target_volume': np.array([1 - random_control_volume]),
            # not much point sampling invalid states
            'control_volume': np.array([random_control_volume]),
            'num_diverged': np.zeros(1, dtype=np.float64),
            'error': np.zeros(1, dtype=np.float64),
        }

        self.scenario_ompl.numpy_to_ompl_state(state_np, state_out)

        if self.plot:
            self.scenario_ompl.s.plot_sampled_state(state_np)


class WateringGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario_ompl: WateringOmpl,
                 rng: np.random.RandomState,
                 target_container_pos: np.ndarray,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(WateringGoalRegion, self).__init__(si)
        self._target_container_pos = target_container_pos
        self.setThreshold(threshold)
        self.goal = goal
        self.midpoint_range_volume = np.mean(self.goal["goal_target_volume_range"]).item()
        self.scenario_ompl = scenario_ompl
        self.rng = rng
        self.plot = plot

    def distanceGoal(self, state: ob.CompoundState):
        """
        Uses the distance between a specific point in a specific subspace and the goal point
        """
        state_np = self.scenario_ompl.ompl_state_to_numpy(state)
        distance = self.scenario_ompl.s.distance_to_goal(state_np, self.goal)

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def sampleGoal(self, state_out: ob.CompoundState):
        sampler = self.getSpaceInformation().allocStateSampler()
        # sample a random state via the state space sampler, in hopes that OMPL will clean up the memory...
        sampler.sampleUniform(state_out)
        random_x_left = self.rng.uniform(low=-0.1, high=0.1)
        random_height = self.rng.uniform(low=0.08, high=0.5)

        goal_state_np = {
            'controlled_container_pos': self._target_container_pos + np.array([random_x_left, random_height]),
            'controlled_container_angle': np.array([0]),
            'target_container_pos': self._target_container_pos,
            'target_volume': np.array([1.0]),  # in practice quite binary
            'control_volume': np.array([0]),
            'error': np.zeros(1, dtype=np.float64),
            'num_diverged': np.zeros(1, dtype=np.float64),
        }
        self.scenario_ompl.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario_ompl.s.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 300
