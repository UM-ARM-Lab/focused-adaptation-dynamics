from unittest import TestCase
import time
import numpy as np
from link_bot_pycommon.water_scenario import WaterSimScenario


class Test(TestCase):
    @classmethod
    def setUpClass(cls):
        params = {
            "gui":                    0,
            "run_flex":               True,
            "save_cfg":               {
                "save_frames": 0,
                "img_size":    64
            },
            "k_pos":                  1.5,
            "k_angle":                2.0,
            "controller_max_horizon": 80
        }
        scenario = WaterSimScenario(params)
        scenario.on_before_data_collection({})
        cls.scenario = scenario  # I assume this is single-threaded...

    def setUp(self):
        self._pos_delta = 0.02
        self._angle_delta = 0.05
        self.scenario.reset()

    @classmethod
    def tearDownClass(cls):
        cls.scenario._scene.close()

    def test_make_toy_scenario(self):
        state = self.scenario.get_state()
        assert state is not None

    def _move_delta(self, delta_pos, delta_angle):
        state = self.scenario.get_state()
        current_controlled_container_pos = state["controlled_container_pos"]
        current_controller_container_angle = state["controlled_container_angle"]
        move_action = {"controlled_container_target_pos":   current_controlled_container_pos + delta_pos,
                       "controlled_container_target_angle": current_controller_container_angle + delta_angle}
        self.scenario.execute_action(None, state, move_action)
        return move_action

    def _assert_delta_action_close(self, delta_pos, delta_angle):
        move_action = self._move_delta(delta_pos, delta_angle)
        self._assert_state_close_to_action(move_action)


    def _assert_state_close_to_action(self, move_action):
        state = self.scenario.get_state()
        for i in range(2):
            self.assertAlmostEqual(state["controlled_container_pos"][i],
                                   move_action["controlled_container_target_pos"][i], delta=self._pos_delta)
        self.assertAlmostEqual(state["controlled_container_angle"], move_action["controlled_container_target_angle"],
                               delta=self._angle_delta)

    def test_move_z_pos(self):
        delta_pos = np.array([0.0, 0.2])
        delta_angle = 0
        self._assert_delta_action_close(delta_pos, delta_angle)
        delta_pos = np.array([0.0, -0.08])
        self._assert_delta_action_close(delta_pos, delta_angle)
        self._assert_has_initial_volumes()

    def _assert_traj_close(self, traj, t):
        state = self.scenario.get_state()
        state_keys = ["controlled_container_pos", "controlled_container_angle", "target_volume", "control_volume"]
        test_state = {}
        for state_key in state_keys: 
            test_state[state_key] = traj[state_key][t]
        self._volume_tol = 0.1
        for i in range(2):
            self.assertAlmostEqual(state["controlled_container_pos"][i],
                                   test_state["controlled_container_pos"][i], delta=self._pos_delta)
        self.assertAlmostEqual(state["controlled_container_angle"], test_state["controlled_container_angle"],
                               delta=self._angle_delta)
        self.assertAlmostEqual(state["control_volume"], test_state["control_volume"],
                               delta=self._volume_tol)
        self.assertAlmostEqual(state["target_volume"], test_state["target_volume"],
                               delta=self._volume_tol)
        


    def test_replay_traj(self):
        traj = np.load("data/low_weight_trajs.npy", allow_pickle=True)[0]
        #First go to the original position
        first_position = traj["controlled_container_pos"][0]
        first_angle = traj["controlled_container_angle"][0]
        state = self.scenario.get_state()
        move_action = {"controlled_container_target_pos":   first_position,
                       "controlled_container_target_angle": first_angle}
        self.scenario.execute_action(None, state, move_action)
        self._assert_state_close_to_action(move_action)

        for i in range(len(traj["controlled_container_target_pos"])):
            state = self.scenario.get_state()
            move_action = {"controlled_container_target_pos":   traj["controlled_container_target_pos"][i],
                           "controlled_container_target_angle": traj["controlled_container_target_angle"][i]}
            self.scenario.execute_action(None, state, move_action)
            self._assert_traj_close(traj, i+1)




    def test_move_x_pos(self):
        delta_pos = np.array([0.05, 0.0])
        delta_angle = 0
        self._assert_delta_action_close(delta_pos, delta_angle)
        delta_pos = np.array([-0.1, 0.0])
        self._assert_delta_action_close(delta_pos, delta_angle)

    def test_move_angle(self):
        delta_pos = np.array([0, 0])
        delta_angle = 0.1
        for i in range(5):
            self._move_delta(np.array([0, 0.1]), 0)
        self._assert_delta_action_close(delta_pos, delta_angle)
        delta_angle = -0.2
        self._assert_delta_action_close(delta_pos, delta_angle)

    def _assert_has_initial_volumes(self):
        state = self.scenario.get_state()
        self.assertAlmostEqual(state["control_volume"], 1, delta=0.02)
        self.assertAlmostEqual(state["target_volume"], 0, delta=0.02)

    def test_dump_on_floor(self):
        self._assert_has_initial_volumes()
        for i in range(5):
            self._move_delta(np.array([0, 0.2]), 0)
        for i in range(4):
            self._move_delta(np.array([0, 0]), 0.7)
        state = self.scenario.get_state()
        self.assertLess(state["control_volume"], 0.1)

    def test_hardcoded_pour_traj(self):
        # This one might need to be updates with dynamics/geometry changes
        self._assert_has_initial_volumes()
        for i in range(5):
            self._move_delta(np.array([0.05, 0.1]), 0)
        for i in range(4):
            self._move_delta(np.array([0, 0]), 0.7)
        state = self.scenario.get_state()
        self.assertLess(state["control_volume"], 0.1)
        self.assertGreater(state["target_volume"], 0.6)
