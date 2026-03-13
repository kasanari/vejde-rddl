import json
import pathlib
import random
import sys
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from regawa import (
    ActionMode,
    GNNParams,
    GraphAgent,
    Grounding,
    GroundingRange,
    agent_from_model,
)
from regawa.data.data import heterostatedata_from_obslist
from regawa.data.torch import heterostatedata_to_tensors
from regawa.model import BaseModel
from regawa.policy.recurrent_gnn_agent import RecurrentGraphAgent
from regawa.rl.util import calc_loss, update

# from regawa.wrappers import fn_objects_with_type
from regawa.wrappers import (
    create_render_graph,
    fn_groundobs_to_heterograph,
    fn_idx_obs,
    from_dict_action,
    object_list,
    remove_false,
)
from regawa.wrappers.grounding_utils import fn_objects_with_type
from tqdm import tqdm

from rddlgraphwrapper.src.regawa.model.null import NullConst
from vejde_rddl import register_env, register_pomdp_env
from vejde_rddl.rddl_utils import rddl_ground_to_tuple

RecordingObs = dict[str, Any]
RecordingAction = dict[str, int]
RecordingEntry = dict[RecordingObs, RecordingAction]
Recording = list[RecordingEntry]

GroundAction = dict[Grounding, bool]
GroundObs = dict[Grounding, Any]

IndexedAction = tuple[int, ...]


@th.inference_mode()
def save_sorted_losses(
    model: BaseModel,
    agent: GraphAgent,
    expert_actions: list[GroundAction],
    indexed_expert_obs: list[GroundObs],
    expert_obs: list[GroundObs],
    device: str = "cpu",
):
    loss_per_obs = []
    objects_with_type = fn_objects_with_type(model.fluent_param)
    for i, (expert_a, d, o) in enumerate(
        zip(expert_actions, indexed_expert_obs, expert_obs, strict=False)
    ):
        s = heterostatedata_to_tensors(heterostatedata_from_obslist([d]), device=device)
        g = to_graph(o, model)
        actions, logprob, _, _, p_a, p_n__a = agent.sample(s, deterministic=True)
        l2_norms = [th.sum(th.square(w)) for w in agent.parameters()]
        loss = calc_loss(l2_norms, logprob).item()

        objs = object_list(list(o.keys()), objects_with_type)
        objs = [o.name for o in objs]

        model_a = from_index_action(actions[0], lambda x: objs[x], model)

        factor_weights = p_n__a.T[actions[:, 0]].detach().squeeze().cpu().numpy()

        weight_by_factor = {
            k: f"{float(v):0.3f}"
            for k, v in zip(g.factor_labels, factor_weights, strict=False)
            if v > 0.001
        }

        weight_by_action = {
            k: f"{float(v):0.3f}"
            for k, v in zip(
                model.action_fluents,
                p_a.detach().squeeze().cpu().numpy(),
                strict=False,
            )
            if v > 0.001
        }

        loss_per_obs.append(
            dict(
                model_action=model_a,
                expert_action=list(expert_a.keys())[0],
                loss=loss,
                action_probs=weight_by_action,
                object_probs=weight_by_factor,
                step=i,
                # obs=o,
            )
        )

    sorted_loss = sorted(loss_per_obs, key=lambda x: x["loss"], reverse=True)

    return sorted_loss


def ground_to_tuple(s: str) -> Grounding:
    return rddl_ground_to_tuple(s)


def convert_state_to_tuples(
    d: RecordingObs, converter_func: Callable[[str], Grounding]
) -> dict[Grounding, Any]:
    return {converter_func(k): v for k, v in d.items()}


def convert_actions_to_tuples(
    d: RecordingAction, converter_func: Callable[[str], Grounding]
) -> dict[Grounding, bool]:
    return (
        convert_state_to_tuples(d, converter_func)
        if d
        else {(NullConst.id, NullConst.type): True}
    )


def ensure_tuple(x: tuple[str, ...]) -> tuple[str, ...]:
    return x + (NullConst.id,) if len(x) == 1 else x


def from_index_action(
    action: IndexedAction, idx_to_obj: Callable[[int], str], model: BaseModel
) -> tuple[str, str]:
    return (model.action_fluents[action[0]], idx_to_obj(action[1]))


def fn_to_indexed_action(model: BaseModel):
    def to_indexed_action(
        action: GroundAction,
        obj_to_idx: Callable[[str], int],
    ) -> IndexedAction:
        action = list(action.keys())[0] if action else (NullConst.id, NullConst.type)
        a = from_dict_action(
            action, lambda x: model.action_fluents.index(x), obj_to_idx
        )
        return a

    return to_indexed_action


render_index = 0


def get_actions(data: Recording):
    return [[x["actions"] for x in d] for d in data]


def get_obs(data: Recording):
    return [[x["state"] for x in d] for d in data]


def to_graph(obs: GroundObs, model: BaseModel):
    create_graph = fn_groundobs_to_heterograph(model)
    g = create_graph(obs)
    return create_render_graph(g.boolean, g.numeric)


def fn_to_obsdata(model: BaseModel):
    create_graphs = fn_groundobs_to_heterograph(model)
    g_to_obsdata = fn_idx_obs(model)
    to_indexed_action = fn_to_indexed_action(model)

    def to_obsdata(
        obs: GroundObs,
        action: GroundAction,
    ):
        g = create_graphs(obs)
        o = g_to_obsdata(g)
        a = to_indexed_action(action, lambda x: g.boolean.factors.index(x))

        # Rendering
        # dot = to_graphviz(create_render_graph(g.boolean, g.numeric))
        # global render_index
        # render_path = pathlib.Path("saved_render")
        # render_path.mkdir(exist_ok=True)
        # with open(render_path / f"graph_{render_index}.dot", "w") as f:
        #     f.write(dot)
        # render_index += 1

        return o, a

    return to_obsdata


def get_rnn_agent(
    model: BaseModel, layers: int, embedding_dim: int, device: str = "cpu"
):
    params = GNNParams(
        layers=layers,
        embedding_dim=embedding_dim,
        activation=th.nn.Mish(),
        aggregation="sum",
        action_mode=ActionMode.NODE_THEN_ACTION,
    )

    return agent_from_model(
        RecurrentGraphAgent,
        model,
        params,
        device=device,
    )


def get_agent(model: BaseModel, layers: int, embedding_dim: int, device: str = "cpu"):
    params = GNNParams(
        layers=layers,
        embedding_dim=embedding_dim,
        activation=th.nn.Tanh(),
        aggregation="max",
        action_mode=ActionMode.NODE_THEN_ACTION,
    )

    return agent_from_model(
        GraphAgent,
        model,
        params,
        device=device,
    )


# def get_rddl_data(data: Recording, model: BaseModel, grounded_model: BaseGroundedModel):
#     create_graphs = create_graphs_func(model)
#     data = [convert_episode(d) for d in data]
#     rollout = [to_obsdata(s, model, grounded_model) for e in data for s in e]
#     return zip(*rollout)


def train_mimic(domain: str, data_path: str, batch_id: str):
    datafile = Path(data_path).expanduser()
    seed = 1
    instance = "1"
    shuffle_batch = True
    device = "cuda:0" if th.cuda.is_available() else "cpu"
    use_rnn = False
    env_id = (
        register_pomdp_env(domain=domain, instance=instance, remove_false=True)
        if use_rnn
        else register_env(domain=domain, instance=instance, remove_false=True)
    )

    run_id = str(uuid.uuid4())

    learning_rate = 1e-3
    wd = 0.0
    num_epochs = 5000
    embedding_dim = 16
    layers = 4
    batch_size = 128

    config = {
        "learning_rate": learning_rate,
        "weight_decay": wd,
        "num_epochs": num_epochs,
        "seed": seed,
        "data_path": data_path,
        "embedding_dim": embedding_dim,
        "shuffle_batch": shuffle_batch,
        "layers": layers,
        "batch_size": batch_size,
    }

    output_dir = pathlib.Path("imitation_output")
    output_dir.mkdir(exist_ok=True)
    batch_dir = output_dir / batch_id
    batch_dir.mkdir(exist_ok=True)
    run_dir = batch_dir / run_id
    run_dir.mkdir()

    env: gym.Env = gym.make(env_id)
    model: BaseModel = env.unwrapped.model
    assert isinstance(model, BaseModel)

    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)

    agent = (
        get_agent(model, layers, embedding_dim, device)
        if not use_rnn
        else get_rnn_agent(model, layers, embedding_dim, device)
    )
    optimizer = th.optim.AdamW(
        agent.parameters(), lr=learning_rate, amsgrad=True, weight_decay=wd
    )

    with open(datafile) as f:
        expert_data = json.load(f)

    expert_actions = [x["actions"] for x in expert_data]
    expert_obs = [x["state"] for x in expert_data]

    def to_tuple(x: str) -> tuple[str, ...]:
        return tuple(x.split("__"))

    def wrapper_func(x: RecordingObs) -> Mapping[Grounding, GroundingRange]:
        return remove_false(
            convert_state_to_tuples(
                x,
                to_tuple,
            )
        )

    expert_actions = [convert_actions_to_tuples(e, to_tuple) for e in expert_actions]
    expert_actions = [
        {ensure_tuple(k): v} for e in expert_actions for k, v in e.items()
    ]
    expert_obs = [wrapper_func(e) for e in expert_obs]

    to_obsdata = fn_to_obsdata(model)

    indexed_expert_obs, indexed_expert_action = zip(
        *[to_obsdata(o, a) for o, a in zip(expert_obs, expert_actions, strict=False)],
        strict=False,
    )

    indexed_expert_action = th.as_tensor(
        indexed_expert_action, dtype=th.int64, device=device
    )
    avg_loss = 0.0
    avg_grad_norm = 0.0
    pbar = tqdm()
    grad_norms = deque()
    losses = deque()

    for _ in range(num_epochs):
        if shuffle_batch:
            perm = th.randperm(len(expert_obs))
            indexed_expert_obs = [indexed_expert_obs[i] for i in perm]
            indexed_expert_action = indexed_expert_action[perm]
            expert_obs = [expert_obs[i] for i in perm]

        if num_epochs > 2000:
            # set the learning rate to 1e-4 after 2000 steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-4

        for i in range(0, len(indexed_expert_obs), batch_size):
            obs_minibatch = indexed_expert_obs[i : i + batch_size]
            action_minibatch = indexed_expert_action[i : i + batch_size]
            d = heterostatedata_to_tensors(
                heterostatedata_from_obslist(obs_minibatch), device=device
            )

            loss, grad_norm, _ = update(
                agent, optimizer, action_minibatch, d, max_grad_norm=0.5
            )
        avg_loss = avg_loss + (loss - avg_loss) / 2
        avg_grad_norm = avg_grad_norm + (grad_norm - avg_grad_norm) / 2
        grad_norms.append(grad_norm)
        losses.append(loss)
        pbar.set_description(f"Loss: {avg_loss:.3f}, Grad Norm: {avg_loss:.3f}")
        pbar.update(1)

    pbar.close()

    agent_path = str(run_dir / f"model_{run_id}.pth")
    config_path = str(run_dir / f"config_{run_id}.json")
    agent.save_agent(agent_path)
    with open(config_path, "w") as f:
        json.dump(config, f)

    instances = range(1, 11)
    instance_returns = []

    for instance in tqdm(instances, total=10):
        _, h = evaluate_instance(
            env_id, domain, instance, agent, True, 100, verbose=False
        )
        instance_returns.append(list(h))

    stats = EvalEntry(
        batch_id, run_id=run_id, domain=domain, instance_returns=instance_returns
    )
    return stats, agent_path, config_path


def main():
    data_path = sys.argv[1]
    domain = sys.argv[2]
    batch_id = sys.argv[3]
    instances = json.loads(sys.argv[4])
    stats, apath, config_path = train_mimic(domain, data_path, batch_id)
    to_print = asdict(stats)
    to_print["agent_path"] = apath
    to_print["config_path"] = config_path
    to_print["train_instances"] = instances

    print(json.dumps(to_print))


if __name__ == "__main__":
    main()
