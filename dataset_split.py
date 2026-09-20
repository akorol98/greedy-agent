import os
import numpy as np
import grid2op

env_name = "l2rpn_icaps_2021_large"
env = grid2op.make(env_name)

seed = 42
rng = np.random.default_rng(seed)

# Get chronic / time-series names
chron_paths = env.chronics_handler.subpaths
chron_names = np.array([os.path.basename(p) for p in chron_paths])

# shuffle
rng.shuffle(chron_names)

n = len(chron_names)
n_val = int(0.10 * n)
n_test = int(0.10 * n)

val_chronics = chron_names[:n_val].tolist()
test_chronics = chron_names[n_val:n_val + n_test].tolist()

nm_train, nm_val, nm_test = env.train_val_split(
    val_scen_id=val_chronics,
    test_scen_id=test_chronics,
    add_for_train="train",
    add_for_val="val",
    add_for_test="test",
)

env_train = grid2op.make(nm_train)
env_val = grid2op.make(nm_val)
env_test = grid2op.make(nm_test)
