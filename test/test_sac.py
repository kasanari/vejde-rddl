from regawa import ActionMode, GNNParams
from regawa.rl.sac_gnn import SACArgs, train
from torch import nn
from vejde_rddl import register_env


def main():
    env_id = register_env(
        domain="SysAdmin_MDP_ippc2011",
        instance="1",
        remove_false=True,
    )
    args = SACArgs(
        agent_class="GraphAgent",
        env_id=env_id,
        total_timesteps=100000,
        target_network_frequency=3000,
        num_envs=2,
        weight_decay=0.0,
        buffer_size=5000,
        learning_starts=1000,
        autotune=True,
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
    agent.save_agent("sac_sysadmin_ippc.pth")


if __name__ == "__main__":
    main()
