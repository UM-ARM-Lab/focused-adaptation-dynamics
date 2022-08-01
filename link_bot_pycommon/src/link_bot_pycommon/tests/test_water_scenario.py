from unittest import TestCase
from link_bot_pycommon.water_scenario import WaterSimScenario

class Test(TestCase):
    def test_make_toy_scenario(self):
        params = {}
        scenario = WaterSimScenario(params)
        state = scenario.get_state()
        assert state is not None


