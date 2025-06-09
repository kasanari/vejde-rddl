from regawa.rl.ppo_gnn import setup, Args
from regawa import GNNParams, ActionMode
from torch import nn
from vejde_rddl import register_env


num_steps = 23
num_envs = 2
distributed_steps = num_steps // num_envs
print(f"Distributed steps: {distributed_steps}")

env_id = register_env()
args = Args(
    env_id=env_id,
    total_timesteps=100_000,
    num_steps=distributed_steps,
    domain="rddl/conditional_bandit/domain.rddl",
    instance="rddl/conditional_bandit/instance_1.rddl",
    # eval_instance="rddl/conditional_bandit/instance_2.rddl",
    weight_decay=0.0,
    remove_false=True,
    multiprocess=False,
    debug=True,
    num_envs=num_envs,
    agent_config=GNNParams(
        layers=4,
        embedding_dim=16,
        activation=nn.Tanh(),
        aggregation="max",
        action_mode=ActionMode.ACTION_THEN_NODE,
    ),
)
setup(args)
