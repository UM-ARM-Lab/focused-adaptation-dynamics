import numpy as np
from ompl import control as oc

from link_bot_agent.my_directed_control_sampler import MyDirectedControlSampler


class GurobiDirectedControlSampler(MyDirectedControlSampler):

    def __init__(self, si, gurobi_solver):
        super(GurobiDirectedControlSampler, self).__init__(si, gurobi_solver, "Gurobi")

    def sampleTo(self, sampler, control, state, target):
        M = self.si.getStateDimension()
        L = self.si.getControlSpace().getDimension()
        o = np.ndarray((M, 1))
        og = np.ndarray((M, 1))
        for i in range(M):
            o[i, 0] = state[i]
            og[i, 0] = target[i]
        u, o_next = self.action_selector.act(o, og)
        for i in range(L):
            control[i] = u[0, 0, i]
        for i in range(M):
            target[i] = o_next[i, 0]
        duration_steps = 1

        GurobiDirectedControlSampler.states_sampled_at.append(state)

        return duration_steps
