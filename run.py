import json
import grid2op
from grid2op.Opponent import BaseOpponent
from lightsim2grid import LightSimBackend
from agent import GreedyAgent


FILDERED_ACTIONS_PATH = "action_spaces/l2rpn_icaps_2021_large/top_actions_1000.json"
WITH_OPPONENT = False

if WITH_OPPONENT:
    env = grid2op.make("l2rpn_icaps_2021_large_train", backend=LightSimBackend())
else:
    env = grid2op.make("l2rpn_icaps_2021_large_train", backend=LightSimBackend(), opponent_class=BaseOpponent)
chronics = env.chronics_handler.available_chronics()
print("Env:", env.name)
print("Lines:", env.n_line)
print("Substations:", env.n_sub)
print("Generators:", env.n_gen)
print("Loads:", env.n_load)
print("Storage:", env.n_storage)
print("max steps in chronic:", env.chronics_handler.real_data.max_iter)
print("Available chronics:", len(chronics))

with open(FILDERED_ACTIONS_PATH, "rt", encoding="utf-8") as action_set_file:
    filtered_actions = [
        env.action_space(a) for a in json.load(action_set_file)
    ]

agent = GreedyAgent(env.action_space, filtered_actions)

done = False
all_obs = []
all_actions = []
env.set_id(chronics[0])
obs = env.reset()
print("Runing chronic:", env.chronics_handler.get_id())

while not done:
    action = agent.act(obs)
    all_obs.append(obs)
    all_actions.append(action)
    obs, reward, done, info = env.step(action)
    print(f"Step: {obs.current_step}, max rho: {obs.rho.max():.4f}")
    
