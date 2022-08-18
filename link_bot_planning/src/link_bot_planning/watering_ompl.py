import warnings
from typing import Dict

import numpy as np

from link_bot_pycommon.water_scenario import WaterSimScenario
from link_bot_pycommon.scenario_ompl import ScenarioOmpl

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.base as ob
    import ompl.control as oc


class WateringOmpl(ScenarioOmpl):

    def __init__(self, scenario: WaterSimScenario, *args, **kwargs):
        super().__init__(scenario, *args, **kwargs)

    @staticmethod
    def numpy_to_ompl_state(state_np: Dict, state_out: ob.CompoundState):
        for i in range(2):
            state_out[0][i] = np.float64(state_np['controlled_container_pos'][i])
        for i in range(2):
            state_out[1][i] = np.float64(state_np['target_container_pos'][i])
        state_out[2][0] = np.float64(state_np['controlled_container_angle'][0])
        state_out[3][0] = np.float64(state_np['control_volume'][0])
        state_out[4][0] = np.float64(state_np['target_volume'][0])

    @staticmethod
    def ompl_state_to_numpy(ompl_state: ob.CompoundState):
        return {
            'controlled_container_pos':   np.array([ompl_state[0][0], ompl_state[0][1]]),
            'target_container_pos':       np.array([ompl_state[1][0], ompl_state[1][1]]),
            'controlled_container_angle': np.array([ompl_state[2][0]]),
            'control_volume':             np.array([ompl_state[3][0]]),
            'target_volume':              np.array([ompl_state[3][0]]),
            'target_container_pos':       np.array([0, 0.1]),
            'controlled_container_angle': np.array([0]),
            'control_volume':             np.array([0]),
            'target_volume':              np.array([0]),
            'num_diverged' : 0
        }

    def ompl_control_to_numpy(self, ompl_state: ob.CompoundState, ompl_control: oc.CompoundControl):
        state_np = self.ompl_state_to_numpy(ompl_state)
        current_pos = state_np['controlled_container_pos']
        current_angle = state_np['controlled_container_angle']
        delta_pos = np.array([ompl_control[0][0], ompl_control[0][1]])
        delta_angle = np.array(ompl_control[1])
        target_pos = current_pos + delta_pos
        target_angle = current_angle + delta_angle
        return {
            'controlled_container_target_pos': target_pos,
            'controlled_container_target_angle': target_angle,
            'controlled_container_target_angle': [0*target_pos[0]],
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
                                      plot=plot)

    def print_oob(self, state: Dict):
        print(state)

    def make_state_space(self):
        state_space = ob.CompoundStateSpace()

        min_x, max_x, min_y, max_y, min_z, max_z = self.planner_params['extent']

        controlled_container_subspace = ob.RealVectorStateSpace(3)
        controlled_container_bounds = ob.RealVectorBounds(3)
        # these bounds are not used for sampling
        controlled_container_bounds.setLow(0, min_x)
        controlled_container_bounds.setHigh(0, max_x)
        controlled_container_bounds.setLow(1, min_y)
        controlled_container_bounds.setHigh(1, max_y)
        controlled_container_bounds.setLow(2, min_z)
        controlled_container_bounds.setHigh(2, max_z)
        controlled_container_subspace.setBounds(controlled_container_bounds)
        controlled_container_subspace.setName("controlled_container")
        state_space.addSubspace(controlled_container_subspace, weight=1)

        if False:
            # extra subspace component for the number of diverged steps
            num_diverged_subspace = ob.RealVectorStateSpace(1)
            num_diverged_bounds = ob.RealVectorBounds(1)
            num_diverged_bounds.setLow(-1000)
            num_diverged_bounds.setHigh(1000)
            num_diverged_subspace.setBounds(num_diverged_bounds)
            num_diverged_subspace.setName("stdev")
            state_space.addSubspace(num_diverged_subspace, weight=0)

        def _state_sampler_allocator(state_space):
            return WateringStateSampler(state_space,
                                            scenario_ompl=self,
                                            extent=self.planner_params['extent'],
                                            rng=self.state_sampler_rng,
                                            plot=self.plot)

        state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(_state_sampler_allocator))

        return state_space

    def make_control_space(self):
        control_space = oc.CompoundControlSpace(self.state_space)

        controlled_container_control_space = oc.RealVectorControlSpace(self.state_space, 2)
        controlled_container_control_bounds = ob.RealVectorBounds(2)
        controlled_container_control_bounds.setLow(0, -0.15)
        controlled_container_control_bounds.setHigh(0, 0.15)
        controlled_container_control_bounds.setLow(1, -0.15)
        controlled_container_control_bounds.setHigh(1, 0.15)
        controlled_container_control_space.setBounds(controlled_container_control_bounds)
        control_space.addSubspace(controlled_container_control_space)

        def _allocator(cs):
            return WateringControlSampler(cs,
                                              scenario_ompl=self,
                                              rng=self.control_sampler_rng,
                                              action_params=self.action_params)

        # Copied from floating_rope_ompl:
        # Peter: I override the sampler here so I can use numpy RNG to make things more deterministic.
        # ompl does not allow resetting of seeds, which causes problems when evaluating multiple
        # planning queries in a row.
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
        self.rng = rng        # extra subspace component for the number of diverged steps
        self.control_space = control_space
        self.action_params = action_params

    def sampleNext(self, control_out, previous_control, state):
        control_out[0][0] = self.rng.uniform(-0.06, 0.06)
        control_out[0][1] = self.rng.uniform(-0.01, 0.06)

    def sampleStepCount(self, min_steps, max_steps):
        step_count = self.rng.randint(min_steps, max_steps)
        return step_count


class WateringStateSampler(ob.RealVectorStateSampler):

    def __init__(self,
                 state_space,
                 scenario_ompl: WateringOmpl,
                 extent,
                 rng: np.random.RandomState,
                 plot: bool):
        super().__init__(state_space)
        self.scenario_ompl = scenario_ompl
        self.rng = rng
        self.plot = plot

    def sampleUniform(self, state_out: ob.CompoundState):
        random_point_x = self.rng.uniform(-0.1, 0.8)
        random_point_y = self.rng.uniform(0.05, 0.8)
        state_np = {
            'controlled_container_pos':      np.array([random_point_x, random_point_y]),
        }

        self.scenario_ompl.numpy_to_ompl_state(state_np, state_out)

        if self.plot:
            self.scenario_ompl.s.plot_sampled_state(state_np)


class WateringGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario_ompl: WateringOmpl,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(WateringGoalRegion, self).__init__(si)
        self.setThreshold(threshold)
        self.goal = goal
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

        goal_state_np = {
            'controlled_container_pos':      self.goal['controlled_container_pos'],
        }

        self.scenario_ompl.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario_ompl.s.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100
