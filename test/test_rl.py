from regawa.rl import train, Args
from regawa import GNNParams, ActionMode
from torch import nn
from vejde_rddl import register_env

def main():
    env_id = register_env()
    args = Args(
        env_id=env_id,
        total_timesteps=2000,
        num_steps=20,
        domain="rddl/conditional_bandit/domain.rddl",
        instance="rddl/conditional_bandit/instance_1.rddl",
        weight_decay=0.0,
        remove_false=True,
        debug=True,
        agent_config=GNNParams(
            layers=4,
            embedding_dim=16,
            activation=nn.Tanh(),
            aggregation="max",
            action_mode=ActionMode.ACTION_THEN_NODE,
        ),
    )
    train(args)
