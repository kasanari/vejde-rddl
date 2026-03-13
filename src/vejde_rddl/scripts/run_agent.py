from pprint import pprint

import tyro
from regawa import GraphAgent, load_agent
from regawa.inference import fn_node_then_action


def main(agent_path: str, rddl_domain: str, rddl_instance: str):
    import gymnasium as gym

    from vejde_rddl import register_env

    agent_class = GraphAgent

    agent, config = load_agent(agent_class, agent_path)

    env: gym.Env = gym.make(
        register_env(
            domain=rddl_domain,
            instance=rddl_instance,
            remove_false=True,
        )
    )
    agent_output = fn_node_then_action(agent, env.unwrapped.model)

    obs, info = env.reset(seed=22)
    done = False
    time = 0
    sum_reward = 0.0
    while not done:
        out = agent_output(obs, info["state"])
        action, *_ = agent.sample_from_obs(obs, deterministic=False)
        obs, reward, terminated, truncated, new_info = env.step(
            action.squeeze(0).tolist()
        )
        sum_reward += float(reward)
        done = terminated or truncated
        time += 1
        print(f"Step: {time}, Reward: {reward}, Sum Reward: {sum_reward}")
        # pprint(info["rddl_state"])
        # #pprint(info["rddl_obs"])
        pprint(out.weight_by_action["e0"])
        pprint(out.weight_by_object)
        pprint(new_info["rddl_action"])
        info = new_info
        # print("-----")


def cli():
    tyro.cli(main)


if __name__ == "__main__":
    cli()
