import numpy as np
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from grid2op.Reward import BaseReward, L2RPNReward
from grid2op.dtypes import dt_float


class MazeRLReward(BaseReward):
    """
    This reward class implements the reward formulated by Dorfer et al. in their paper (https://arxiv.org/pdf/2211.05612).
    Rewards are bounded in [0,1] and grow with decreasing max rho.
    """
    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.min_reward = 0
        self.max_reward = 1

    def __call__(self, action: BaseAction, env: BaseEnv, has_error: bool, is_done: bool, is_illegal: bool, is_ambiguous: bool) -> float:
        if not is_done and not has_error:
            rho = env.current_obs.rho
            rho_max = max(rho)
            n_offline = env.n_line - sum(env.current_obs.line_status)
            if rho_max <= 1.0:
                u = max(rho_max - 0.5, 0)
            else:
                u = np.sum(rho[rho > 1] - 0.5)

            res = np.exp(-u - 0.5 * n_offline)
        else:
            res = self.reward_min
        return res


class BaseWithBonus(BaseReward):
    def __init__(self):
        super().__init__()
        self.min_reward = -300
        self.max_reward = 500
        self.base_reward = L2RPNReward()

    def __call__(self, action: BaseAction, env: BaseEnv, has_error: bool, is_done: bool, is_illegal: bool, is_ambiguous: bool) -> float:
        if not env.done:
            return self.base_reward(action, env, has_error, is_done, is_illegal, is_ambiguous)
        elif env.max_episode_duration() == env.nb_time_step:
            return 500
        else:
            return -300

class HRL2023Reward(L2RPNReward):
    def initialize(self, env):
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            return -0.5
        else:
            r_margins = super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
            n_line = env.n_line
            return r_margins / n_line

#def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
#    if is_done:
#        return -0.5

#    flows = env.get_obs().a_or
#    limits = env.get_thermal_limit()

#    margins = np.maximum((limits - flows) / limits, 0.0)
#    reward_per_line = 1.0 - (1.0 - margins) ** 2

#    return reward_per_line.mean()