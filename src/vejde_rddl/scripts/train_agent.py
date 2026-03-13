from pathlib import Path

import tyro
from regawa import ActionMode, GNNParams
from regawa.rl import Args, train
from torch import nn

from vejde_rddl import RDDLModel, register_shuffle_env


def main(
    domain: str, instances: list[str], agent_save_path: Path, args: Args | None = None
):
    env_id = register_shuffle_env(
        domain=domain,
        instance=instances,
        remove_false=True,
    )
    if args is None:
        print("No args provided, using default args.")
        args = Args(
            agent_class="GraphAgent",
            env_id=env_id,
            total_timesteps=4000,
            rollout_length=128,
            num_minibatches=2,
            learning_rate=0.001,
            clip_coef=0.3,
            ent_coef=0.1,
            debug=True,
            num_envs=1,
            vf_coef=1.0,
            weight_decay=0.0,
            gae_lambda=0.95,
            update_epochs=10,
            max_grad_norm=1.0,
            agent_config=GNNParams(
                layers=8,
                embedding_dim=16,
                activation=nn.Tanh(),
                aggregation="max",
                action_mode=ActionMode.ACTION_THEN_NODE,
            ),
        )

    stats, agent = train(args)
    agent.save_agent(agent_save_path)
    print(f"Agent saved to {agent_save_path}")
    print(f"Training stats: {stats}")


def cli_main():
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)

if __name__ == "__main__":
    cli_main()
