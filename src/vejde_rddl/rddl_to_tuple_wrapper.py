from functools import cache
from typing import Any

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict
from pyRDDLGym import RDDLEnv

from .rddl_utils import rddl_ground_to_tuple

WrapperObsType = spaces.Dict
WrapperActType = spaces.Tuple


@cache
def merge_rddl_grounding(grounding: tuple[str, ...]):
    action_fluent, *params = grounding
    return f"{action_fluent}___{'__'.join(params)}" if params else action_fluent


class RDDLToTuple(gym.Wrapper[WrapperActType, WrapperObsType, Dict, Dict]):
    def __init__(self, env: RDDLEnv) -> None:
        super().__init__(env)

    def step(
        self,
        actions: Dict,
    ) -> tuple[
        dict[str, bool | None],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        obs, reward, terminated, truncated, info = self.env.step(
            {merge_rddl_grounding(k): v for k, v in actions.items()}
        )

        obs = {rddl_ground_to_tuple(k): v for k, v in obs.items()}

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        obs = {rddl_ground_to_tuple(k): v for k, v in obs.items()}

        return obs, info
