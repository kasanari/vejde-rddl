from pprint import pprint

import tyro
from regawa import RecurrentGraphAgent, load_agent


def main(agent_path: str, rddl_domain: str, rddl_instance: str):
    import gymnasium as gym

    from vejde_rddl import register_pomdp_env as register_env

    agent_class = RecurrentGraphAgent

    agent, config = load_agent(agent_class, agent_path)

    # stacking = True
    add_actions_to_obs = True
    add_initial_state = True
    remove_false = True

    env: gym.Env = gym.make(
        register_env(
            domain=rddl_domain,
            instance=rddl_instance,
            remove_false=remove_false,
            add_actions_to_obs=add_actions_to_obs,
            add_initial_state=add_initial_state,
        )
    )

    obs, info = env.reset(seed=22)
    done = False
    time = 0
    sum_reward = 0.0
    while not done:
        action, *_ = agent.sample_from_obs(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action.squeeze(0).tolist())
        sum_reward += float(reward)
        done = terminated or truncated
        time += 1
        print(f"Step: {time}, Reward: {reward}, Sum Reward: {sum_reward}")
        # print(info["rddl_state"])
        pprint(info["rddl_obs"])
        pprint(info["rddl_action"])
        print("-----")


def cli():
    tyro.cli(main)


if __name__ == "__main__":
    cli()
