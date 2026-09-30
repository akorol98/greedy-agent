from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from grid2op.Agent import BaseAgent
import numpy as np


_worker_obs = None
_worker_actions = None


def _init_sim_worker(obs, actions):
    global _worker_obs, _worker_actions
    _worker_obs = obs
    _worker_actions = actions


def _score_actions(obs, actions, start, stop):
    best_index = None
    best_rho = float("inf")
    for index in range(start, stop):
        sim_obs, _, done, _ = obs.simulate(actions[index])
        rho = sim_obs.rho.max()
        if rho < best_rho and not done:
            best_index = index
            best_rho = rho
    return best_index, best_rho


def _score_action_chunk(bounds):
    return _score_actions(_worker_obs, _worker_actions, *bounds)


class GreedyAgent(BaseAgent):
    def __init__(self, action_space, filtered_actions, revert_thr=0.85, action_thr=0.95,
                 n_workers=8):
        super().__init__(action_space)
        self.filtered_actions = filtered_actions
        self.action_space = action_space
        self.revert_thr = revert_thr
        self.action_thr = action_thr
        self.n_workers = n_workers

    def reset(self, observation):
        pass

    def get_best_action_sim(self, obs, actions):

        n_workers = min(self.n_workers, len(actions))
        if n_workers == 1:
            best_index, _ = _score_actions(obs, actions, 0, len(actions))
            return self.action_space() if best_index is None else actions[best_index]

        chunks = [
            (i * len(actions) // n_workers, (i + 1) * len(actions) // n_workers)
            for i in range(n_workers)
        ]
        best_index = None
        best_rho = float("inf")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("fork"),
            initializer=_init_sim_worker,
            initargs=(obs, actions),
        ) as executor:
            # map preserves action order, including ties across worker chunks.
            for index, rho in executor.map(_score_action_chunk, chunks):
                if index is not None and rho < best_rho:
                    best_index, best_rho = index, rho

        return self.action_space() if best_index is None else actions[best_index]

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

    def act(self, observation, reward=None, done=None):
        reconnect_action = self.get_reconnect_action(observation)

        if observation.rho.max() > self.action_thr:
            best_action = self.get_best_action_sim(observation, self.filtered_actions)
            return best_action + reconnect_action

        if observation.rho.max() < self.revert_thr:
            return self.revert_one_substation(observation) + reconnect_action
        
        return self.action_space() + reconnect_action

if __name__ == "__main__":
    import grid2op
    from grid2op.Agent import BaseAgent
    from grid2op.Opponent import BaseOpponent
    from lightsim2grid import LightSimBackend
    import json
    import time

    FILTERED_ACTIONS = "action_spaces/l2rpn_icaps_2021_large/top_actions_1000.json"
    WITH_OPPONENT = False

    if WITH_OPPONENT:
        env = grid2op.make("l2rpn_icaps_2021_large_val", backend=LightSimBackend())
    else:
        env = grid2op.make("l2rpn_icaps_2021_large_val", backend=LightSimBackend(), opponent_class=BaseOpponent)
        
    with open(FILTERED_ACTIONS, "rt", encoding="utf-8") as action_set_file:
        filtered_actions = [
            env.action_space(a) for a in json.load(action_set_file)
        ]

    agent = GreedyAgent(env.action_space, filtered_actions, n_workers=8)

    obs = env.reset(seed=1)
    done = False
    time_start = time.time()
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        print(f"Step: {env.nb_time_step}, Reward: {reward:.4f}, Max Rho: {obs.rho.max():.4f}")
    
    print(f"Total time taken: {time.time() - time_start:.2f}")