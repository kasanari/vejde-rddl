import gymnasium as gym
import pytest
from vejde_rddl import register_env, register_pomdp_env


@pytest.fixture(scope="module")
def tiger_env():
    env_id = register_env(
        domain="rddl/tiger/domain.rddl",
        instance="rddl/tiger/instance_1.rddl",
        remove_false=True,
    )
    env = gym.make(env_id)
    return env


@pytest.fixture(scope="module")
def stacking_tiger_env():
    env_id = register_pomdp_env(
        domain="rddl/tiger/domain.rddl",
        instance="rddl/tiger/instance_2.rddl",
        remove_false=True,
    )
    env = gym.make(env_id)
    return env


@pytest.mark.parametrize(
    "action, expected_reward",
    [
        (("open", "left"), 10.0),  # open left door
        (("open", "right"), -100.0),  # open right door
    ],
)
def test_open(tiger_env, action, expected_reward):
    obs, info = tiger_env.reset()

    print("Info:", info["rddl_state"])

    predicate = info["action_fluents"].index(action[0])
    door = info["idx_to_object"].index(action[1])

    obs, reward, terminated, truncated, info = tiger_env.step((predicate, door))

    print("Info:", info["rddl_state"])

    assert (
        terminated is True
    ), "The episode should be terminated after opening the door."
    assert (
        reward == expected_reward
    ), f"Expected reward {expected_reward}, but got {reward}."


@pytest.mark.parametrize(
    "action, expect_hear",
    [
        (
            ("listen", "left"),
            False,
        ),  # we do not expect to hear the tiger from the left door
        (
            ("listen", "right"),
            True,
        ),  # we expect to hear the tiger from the right door
    ],
)
def test_listen(tiger_env, action, expect_hear):
    obs, info = tiger_env.reset(seed=22)

    print("Info:", info["rddl_state"])

    predicate = info["action_fluents"].index(action[0])
    door = info["idx_to_object"].index(action[1])

    obs, reward, terminated, truncated, info = tiger_env.step((predicate, door))

    print("Info:", info["rddl_state"])

    assert terminated is False, "The episode should not be terminated after listening."

    if expect_hear:
        assert ("growl", "right") in info[
            "rddl_state"
        ], "Did not expect to hear the tiger."
        assert ("growl", "left") not in info[
            "rddl_state"
        ], "Did not expect to hear the tiger on the left side."
    else:
        assert ("growl", "right") not in info[
            "rddl_state"
        ], "Expected to hear the tiger."
        assert ("growl", "left") not in info[
            "rddl_state"
        ], "Expected to hear the tiger."


def test_false_positive(tiger_env):
    obs, info = tiger_env.reset(seed=22)

    action = (
        "listen",
        "left",
    )  # though the tiger is on the right, we can observe a false positive from the left door if we keep listening
    predicate = info["action_fluents"].index(action[0])
    door = info["idx_to_object"].index(action[1])

    heard_tiger = False
    done = False
    while not done:
        obs, reward, terminated, truncated, info = tiger_env.step((predicate, door))

        assert reward == -1.0
        print("Info:", info["rddl_state"])

        if ("growl", "left") in info["rddl_state"]:
            heard_tiger = True
            break

        done = terminated or truncated

    assert heard_tiger, "Did not hear the tiger from the left door"


def test_stacking_tiger_env(stacking_tiger_env: gym.Env):
    import random

    # action_mode = ActionMode.ACTION_THEN_NODE
    # params = GNNParams(
    #     layers=3,
    #     embedding_dim=16,
    #     activation=th.nn.Mish(),
    #     aggregation="sum",
    #     action_mode=action_mode,
    # )

    # agent = agent_from_env("RecurrentGraphAgent", stacking_tiger_env, params)

    obs, info = stacking_tiger_env.reset(seed=22)
    turns = 100
    total_reward = 0.0
    for t in range(turns):
        doors = info["idx_to_object"]
        if t < turns - 1:
            chosen_door = random.choice(doors[1:])
            action_str = "listen"
        else:
            obs = info["rddl_obs"]
            # on the last turn, open the door we believe the tiger is not behind
            num_left_growls = sum(obs.get(("growl", "left"), []))
            num_right_growls = sum(obs.get(("growl", "right"), []))
            action_str = "open"
            if num_left_growls > num_right_growls:
                chosen_door = "right"
            else:
                chosen_door = "left"
        action = (
            info["action_fluents"].index(action_str),
            info["idx_to_object"].index(chosen_door),
        )
        obs, reward, terminated, truncated, info = stacking_tiger_env.step(action)
        total_reward += reward
    assert terminated is True
    assert truncated is True
    assert chosen_door == "left", "The tiger should be behind the right door."
    print("Reward:", total_reward)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
