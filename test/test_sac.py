from regawa.rl.sac_gnn import train, SACArgs
from regawa import GNNParams, ActionMode
from torch import nn
from vejde_rddl import register_env


def main():
    env_id = register_env(
        domain="rddl/conditional_bandit/domain.rddl",
        instance="rddl/conditional_bandit/instance_1.rddl",
        remove_false=True,
    )
    args = SACArgs(
        agent_class="GraphAgent",
        env_id=env_id,
        total_timesteps=4000,
        target_network_frequency=100,
        num_envs=2,
        weight_decay=0.0,
        buffer_size=128,
        mlflow_tracking_uri="sqlite:///mlruns.db",
        debug=True,
        agent_config=GNNParams(
            layers=4,
            embedding_dim=16,
            activation=nn.Tanh(),
            aggregation="max",
            action_mode=ActionMode.ACTION_THEN_NODE,
        ),
    )
    agent = train(args)


if __name__ == "__main__":
    main()
