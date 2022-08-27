from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.scenario_ompl import ScenarioOmpl


def get_ompl_scenario(scenario: ExperimentScenario, *args, **kwargs) -> ScenarioOmpl:
    # the imports are local to avoid importing all dependencies at all times

    # order matters here, because of inheritance
    from link_bot_pycommon.rope_dragging_scenario import RopeDraggingScenario
    if isinstance(scenario, RopeDraggingScenario):
        from link_bot_planning.rope_dragging_ompl import RopeDraggingOmpl
        return RopeDraggingOmpl(scenario, *args, **kwargs)

    from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
    if isinstance(scenario, BaseDualArmRopeScenario):
        from link_bot_planning.dual_arm_rope_ompl import DualArmRopeOmpl
        return DualArmRopeOmpl(scenario, *args, **kwargs)
    from link_bot_pycommon.floating_rope_scenario import FloatingRopeScenario
    from link_bot_planning.floating_rope_ompl import FloatingRopeOmpl
    if isinstance(scenario, FloatingRopeScenario):
        return FloatingRopeOmpl(scenario, *args, **kwargs)

    from link_bot_pycommon.water_scenario import WaterSimScenario
    if isinstance(scenario, WaterSimScenario):
        from link_bot_planning.watering_ompl import WateringOmpl
        return WateringOmpl(scenario, *args, **kwargs)
    else:
        raise NotImplementedError(f"unimplemented scenario {scenario}")
