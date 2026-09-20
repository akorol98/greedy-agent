from grid2op.Agent import BaseAgent
import numpy as np


class GreedyAgent(BaseAgent):
    def __init__(self, action_space, filtered_actions):
        super().__init__(action_space)
        self.filtered_actions = filtered_actions
        self.action_space = action_space

    def reset(self, observation):
        pass

    def get_best_action_sim(self, obs, actions):
        best_rho = float("inf")
        best_action = None
        
        for action in actions:
            sim_obs, reward, done, info = obs.simulate(action)
                            
            if sim_obs.rho.max() < best_rho and not done:
                best_rho = sim_obs.rho.max()
                best_action = action

        if best_action is None: best_action = self.action_space()
        return best_action

    def revert_one_substation(self, obs):
        pos2 = np.where(obs.topo_vect == 2)[0]

        if len(pos2) == 0:
            return self.action_space()

        subs = set(int(obs._topo_vect_to_sub[p]) for p in pos2)

        best_action = self.action_space()
        best_rho = float(np.max(obs.rho))

        for sub_id in subs:
            if obs.time_before_cooldown_sub[sub_id] > 0:
                continue

            sub_pos = np.where(obs._topo_vect_to_sub == sub_id)[0]
            set_bus = [(int(p), 1) for p in sub_pos if obs.topo_vect[p] == 2]

            if not set_bus:
                continue

            action = self.action_space()
            action.set_bus = set_bus

            sim_obs, reward, done, info = obs.simulate(action)

            if done:
                continue

            if info.get("is_illegal", False) or info.get("is_ambiguous", False):
                continue

            max_rho = float(np.max(sim_obs.rho))

            if not np.isfinite(max_rho):
                continue

            if max_rho < best_rho:
                best_rho = max_rho
                best_action = action
                print(f"Reverting substation {sub_id} max rho sim {best_rho:.4f}")

        return best_action

    def get_reconnect_action(self, obs):
        line_stat_s = obs.line_status
        cooldown = obs.time_before_cooldown_line
        can_be_reco = ~line_stat_s & (cooldown == 0)
        
        action = self.action_space()
        if can_be_reco.any():
            reconnect_line = [
                    self.action_space({"set_line_status": [(id_, +1)]})
                    for id_ in (can_be_reco).nonzero()[0]
                ]
            for line in reconnect_line: action += line
                
            rollout_obs, reward, done, info = obs.simulate(action, time_step=1)
        
            if rollout_obs.rho.max() < 1 and not done:     
                return action
            else:
                return self.action_space()
        else:
            return action

    def act(self, observation):
        reconnect_action = self.get_reconnect_action(observation)

        if observation.rho.max() > 0.95:
            best_action = self.get_best_action_sim(observation, self.filtered_actions)
            return best_action + reconnect_action

        if observation.rho.max() < 0.85:
            return self.revert_one_substation(observation) + reconnect_action
        
        return self.action_space() + reconnect_action