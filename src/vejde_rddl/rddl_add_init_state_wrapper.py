from typing import Any, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyRDDLGym import RDDLEnv
from pyRDDLGym.core.compiler.model import RDDLLiftedModel, RDDLPlanningModel
from pyRDDLGym.core.debug.exception import (
    RDDLInvalidObjectError,
    RDDLRepeatedVariableError,
    RDDLUndefinedVariableError,
)

from .rddl_model import RDDLModel

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = spaces.Dict
WrapperActType = spaces.Dict


def _extract_states(model: RDDLLiftedModel):
    PRIME = RDDLPlanningModel.NEXT_STATE_SYM

    # get the information for each state from the domain
    states, statesranges, nextstates, prevstates = {}, {}, {}, {}
    assert model.ast is not None
    for pvar in model.ast.domain.pvariables:
        if pvar.is_state_fluent():
            name = pvar.name
            statesranges[name] = pvar.range
            nextstates[name] = name + PRIME
            prevstates[name + PRIME] = name
            default = model.variable_defaults[name]
            states[name] = {gname: default for gname in model.variable_groundings[name]}

    # update the state values with the values in the instance
    init_state_info = getattr(model.ast.instance, "init_state", [])
    already_set = {}
    init_state = {}
    for (name, p), value in init_state_info:
        # check whether name is a valid state-fluent
        grounded_states = states.get(name, {})
        if grounded_states is None:
            raise RDDLUndefinedVariableError(
                f"Variable <{name}> referenced in init-state block "
                f"is not a valid state-fluent."
            )

        # extract the grounded name and check that parameters are valid
        params = RDDLPlanningModel.strip_literals(p) if p is not None else ()
        gname = RDDLPlanningModel.ground_var(name, params)
        if gname not in grounded_states:
            required_types = model.variable_params[name]
            raise RDDLInvalidObjectError(
                f"Parameter(s) {params} of state-fluent <{name}> "
                f"declared in the init-state block are not valid, "
                f"must be of type(s) {required_types}."
            )

        # make sure value is correct type
        if isinstance(value, str):
            value = RDDLPlanningModel.strip_literal(value)
            value_type = model.object_to_type.get(value, None)
            required_type = statesranges[name]
            if value_type != required_type:
                if value_type is None:
                    raise RDDLInvalidObjectError(
                        f"State-fluent <{name}> of range <{required_type}> "
                        f"is initialized in init-state block with undefined "
                        f"object <{value}>."
                    )

                raise RDDLInvalidObjectError(
                    f"State-fluent <{name}> of range <{required_type}> "
                    f"is initialized in init-state block with object "
                    f"<{value}> of type <{value_type}>."
                )

        # make sure no duplication
        if gname in already_set and already_set[gname] != value:
            raise RDDLRepeatedVariableError(
                f"Multiple distinct initial values assigned to state-fluent <{gname}> "
                f"in the instance."
            )

        already_set[gname] = value

        init_state[gname] = np.bool_(value) if statesranges[name] == "bool" else value

    return init_state


class RDDLAddInitState(gym.Wrapper[WrapperActType, WrapperObsType, ObsType, ActType]):
    def __init__(self, env: RDDLEnv, only_add_on_reset: bool = False) -> None:
        super().__init__(env)

        rddl_model = RDDLModel(env.unwrapped.model)

        # init_state_info = getattr(rddl_model.model.ast.instance, 'init_state', [])

        self.rddl_model = rddl_model
        self.init_state = _extract_states(rddl_model.model)
        self.only_add_on_reset = only_add_on_reset

    def step(
        self,
        actions: ActType,
    ) -> tuple[
        tuple[dict[str, bool | None], dict[str, bool | None]],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        obs, r, term, trunc, info = self.env.step(actions)
        new_obs = obs if self.only_add_on_reset else self.init_state | obs

        return new_obs, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        new_obs = self.init_state | obs

        return new_obs, info
