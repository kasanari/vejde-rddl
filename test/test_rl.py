from regawa.rl import train, Args
from regawa import GNNParams, ActionMode
from torch import nn
from vejde_rddl import register_env


def main():
    env_id = register_env(
        domain="rddl/conditional_bandit/domain.rddl",
        instance="rddl/conditional_bandit/instance_1.rddl",
        remove_false=True,
    )
    args = Args(
        agent_class="GraphAgent",
        env_id=env_id,
        total_timesteps=2000,
        num_steps=20,
        weight_decay=0.0,
        debug=True,
        agent_config=GNNParams(
            layers=4,
            embedding_dim=16,
            activation=nn.Tanh(),
            aggregation="max",
            action_mode=ActionMode.ACTION_THEN_NODE,
        ),
    )
    stats, _ = train(args)
    print(stats)


if __name__ == "__main__":
    main()
