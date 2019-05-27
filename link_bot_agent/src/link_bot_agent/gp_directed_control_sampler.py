import numpy as np

from link_bot_agent.my_directed_control_sampler import MyDirectedControlSampler


class GPDirectedControlSampler(MyDirectedControlSampler):
    states_sampled_at = []

    def __init__(self, si, gp_model):
        super(GPDirectedControlSampler, self).__init__(si, "random")
        self.gp_model = gp_model

    def sampleTo(self, control_out, previous_control, state, target_out):
        # we return 0 to indicate no duration when LQR gives us a control that takes us into collision
        # this will cause the RRT to throw out this motion
        N = self.si.getStateSpace().getDimension()
        L = self.si.getControlSpace().getDimension()
        s = np.ndarray((N, 1))
        s_target = np.ndarray((N, 1))
        o_k = np.ndarray((P, 1))
        for i in range(M):
            o_d[i, 0] = state[0][i]
            o_d_target[i, 0] = target_out[0][i]
        for i in range(P):
            o_k[i, 0] = state[1][i]

        u, o_d_next, o_k_next = self.action_selector.dual_act(o_d, o_k, o_d_target)

        for i in range(L):
            control_out[i] = u[0, 0, i]
        for i in range(M):
            target_out[0][i] = o_d_next[i, 0]
        for i in range(P):
            target_out[1][i] = o_k_next[i, 0]

        LQRDirectedControlSampler.states_sampled_at.append(state)

        # check validity
        if not self.si.isValid(target_out):
            return 0

        duration_steps = 1
        return duration_steps
