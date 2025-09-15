from regawa.rl import train, Args
from regawa import GNNParams, ActionMode
from torch import nn
from vejde_rddl import register_env
import gymnasium as gym

def main():
    problem = "tiger"
    env_id = register_env(
        domain = f"rddl/{problem}/domain.rddl",
        instance = f"rddl/{problem}/instance_2.rddl",
        remove_false=True,
    )
    args = Args(
        agent_class="RecurrentGraphAgent",
        env_id=env_id,
        total_timesteps=4000,
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
    stats, agent = train(args)

    env = gym.make(env_id)

    obs, info = env.reset(seed=22)
    done = False
    time = 0
    sum_reward = 0
    while not done:
        action, *_ = agent.sample_from_obs(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        sum_reward += reward
        done = terminated or truncated
        time += 1
        print(f"Step {time}: Reward {reward}, Sum Reward {sum_reward}")
    print(f"Total Reward after {time} steps: {sum_reward}")

if __name__ == "__main__":
    main()
