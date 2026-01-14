from copy import copy
from functools import cache, cached_property
from itertools import chain

from pyRDDLGym.core.compiler.model import RDDLLiftedModel  # type: ignore
from regawa import BaseModel
from regawa.model.null import NullConst


class RDDLModel(BaseModel):
    def __init__(self, model: RDDLLiftedModel, add_null_action: bool = True) -> None:
        self.model = model
        self.add_null_action = add_null_action

    @cached_property
    def enum_fluents(self) -> dict[str, list[str]]:
        model = self.model
        fluents = list(
            chain(
                *(
                    model.state_fluents.keys() if model.state_fluents else [],
                    model.non_fluents.keys() if model.non_fluents else [],
                    model.observ_fluents.keys() if model.observ_fluents else [],
                    model.action_fluents.keys() if model.action_fluents else [],
                )
            )
        )

        enum_types = self.model.enum_types if self.model.enum_types else []
        variable_ranges = x if (x := model.variable_ranges) is not None else {}
        return {r: [f for f in fluents if r == variable_ranges[f]] for r in enum_types}

    @cache
    def _enum_values(self, enum: str):
        object_to_type = self.model.object_to_type if self.model.object_to_type else {}
        return [o for o, t in object_to_type.items() if t == enum]

    @cached_property
    def enum_values(self) -> dict[str, list[str]]:
        enum_types = self.model.enum_types if self.model.enum_types else []
        return {t: self._enum_values(t) for t in enum_types}

    @cached_property
    def combined_enum_fluents(self) -> dict[str, str]:
        enum_fluents = self.enum_fluents
        enum_values = self.enum_values

        return {
            f"{f}^^^{i}": f
            for enum, values in enum_values.items()
            for f in enum_fluents[enum]
            for i, v in enumerate(values)
        }

    def combined_enum_fluent(self, fluent: str):
        variable_ranges = x if (x := self.model.variable_ranges) is not None else {}
        enum = variable_ranges[fluent]
        enum_values = self.enum_values[enum]

        return [f"{fluent}^^^{i}" for i, _ in enumerate(enum_values)]

    @cache
    def arity(self, fluent: str) -> int:
        return self.arities[fluent]

    @cache
    def fluents_of_arity(self, arity: int) -> tuple[str, ...]:
        return self._fluents_of_arity[arity]

    @cache
    def idx_to_fluent(self, idx: int) -> str:
        return self.fluents[idx]

    @cache
    def idx_to_type(self, idx: int) -> str:
        return self._idx_to_type[idx]

    @cache
    def fluent_to_idx(self, relation: str) -> int:
        return self._rel_to_idx[relation]

    @cache
    def fluent_params(self, variable: str) -> tuple[str, ...]:
        return self._variable_params[variable]

    @cache
    def fluent_param(self, fluent: str, position: int) -> str:
        """Types/class of the variable/object the fluent/predicate takes as parameter in a given position. Can be seen as the column name in a database table."""
        return self._variable_params[fluent][position]

    @cache
    def fluent_range(self, fluent: str) -> type:
        return self.variable_ranges[fluent]

    @cache
    def idx_to_action(self, idx: int) -> str:
        return self.action_fluents[idx]

    @cache
    def action_to_idx(self, action: str) -> int:
        return self.action_fluents.index(action)

    @cached_property
    def fluents(self) -> tuple[str, ...]:
        x: chain[str] = chain(
            self.model.state_fluents.keys() if self.model.state_fluents else [],
            self.model.non_fluents.keys() if self.model.non_fluents else [],
            self.model.observ_fluents.keys() if self.model.observ_fluents else [],
            self.model.action_fluents.keys() if self.model.action_fluents else [],
        )

        fluents_with_enums = list(chain(*self.enum_fluents.values()))

        y = filter(lambda e: e not in fluents_with_enums, x)

        combined_enum_fluents = self.combined_enum_fluents.keys()

        x = chain(y, combined_enum_fluents)

        return tuple([NullConst.action, *sorted(x)])

    @cached_property
    def types(self) -> tuple[str, ...]:
        return tuple(self._idx_to_type)

    @cached_property
    def _idx_to_type(self) -> list[str]:
        return [NullConst.type, *sorted(set(self._obj_to_type.values()))]

    @cached_property
    def _obj_to_type(self) -> dict[str, str]:
        model: RDDLLiftedModel = self.model
        object_to_type: dict[str, str] = (
            copy(model.object_to_type) if model.object_to_type else {}
        )
        return object_to_type

    @cached_property
    def num_types(self) -> int:
        return len(self._idx_to_type)

    @cached_property
    def variable_ranges(self) -> dict[str, type]:
        mapping = {
            "bool": bool,
            "int": int,
            "real": float,
        }
        enum_types = self.model.enum_types if self.model.enum_types else []
        mapping.update({e: bool for e in enum_types})

        _variable_ranges = x if (x := self.model.variable_ranges) is not None else {}
        variable_ranges: dict[str, type] = {
            key: mapping[value] for key, value in _variable_ranges.items()
        }

        variable_ranges[NullConst.action] = bool

        combined_enum_fluents = {f: bool for f in self.combined_enum_fluents}

        return {
            f: variable_ranges.get(f, combined_enum_fluents.get(f))
            for f in self.fluents
        }

    @cached_property
    def _variable_params(self) -> dict[str, tuple[str, ...]]:
        variable_params: dict[str, list[str]] = copy(self.model.variable_params)  # type: ignore
        variable_params[NullConst.action] = [NullConst.type]

        combined_enum_fluent_params = {
            f: variable_params[v] for f, v in self.combined_enum_fluents.items()
        }

        variable_params = {
            f: variable_params.get(f, combined_enum_fluent_params.get(f))
            for f in self.fluents
        }

        return {k: tuple(v) for k, v in variable_params.items()}

    @cached_property
    def _fluents_of_arity(self) -> dict[int, tuple[str, ...]]:
        arities: dict[str, int] = self.arities
        return {
            value: tuple([k for k, v in arities.items() if v == value])
            for _, value in arities.items()
        }

    @cached_property
    def action_fluents(self) -> tuple[str, ...]:
        model = self.model
        action_fluents = model.action_fluents if model.action_fluents else []

        enum_fluents = list(chain(*self.enum_fluents.values()))

        new_fluents = chain(
            *[self.combined_enum_fluent(f) for f in action_fluents if f in enum_fluents]
        )

        action_fluents = {f for f in action_fluents if f not in enum_fluents}
        action_fluents = action_fluents | set(new_fluents)

        return (
            (NullConst.action, *sorted(action_fluents))
            if self.add_null_action
            else tuple(sorted(action_fluents))
        )  # type: ignore

    @cached_property
    def num_actions(self) -> int:
        return len(self.action_fluents)

    @cached_property
    def num_fluents(self) -> int:
        return len(self.fluents)

    @cache
    def type_to_idx(self, _type: str) -> int:
        return self._type_to_idx[_type]

    @cached_property
    def _type_to_idx(self) -> dict[str, int]:
        return {
            symb: idx for idx, symb in enumerate(self._idx_to_type)
        }  # 0 is reserved for padding

    @cached_property
    def _rel_to_idx(self) -> dict[str, int]:
        return {
            symb: idx for idx, symb in enumerate(self.fluents)
        }  # 0 is reserved for padding

    @cached_property
    def arities(self) -> dict[str, int]:
        return {key: len(value) for key, value in self._variable_params.items()}
