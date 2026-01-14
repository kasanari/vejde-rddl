# Vejde for RDDL

This an RDDL extension for [Vejde](https://github.com/kasanari/vejde).

It wraps [pyRDDLGym](https://github.com/pyrddlgym-project/pyRDDLGym), and provides an child class of BaseModel which automatically pulls the required fields from the simulator.
This lets you experiment with many of the problems in the [library of RDDL problems](https://github.com/pyrddlgym-project/rddlrepository). 

# How do I run it?

If you want a simple sanity check, run `test/test_imitation_mdp.py` which runs a round of supervised learning on a very simple environment using a set policy as the expert.

For quick reinforcement learning, you can use the `train_agent` script:

```bash
uv run train --domain SysAdmin_MDP_ippc2011 --instances 1 2 3
```
