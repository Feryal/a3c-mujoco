a3c-mujoco
=========

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Simulated target reaching tasks using the MuJoCo physics engine. The setup is adapted from [1] end-to-end learning setup for solving pixel-driven control of Jaco arm where learning is accomplished using Asynchronous Advantage Actor-Critic (A3C)[2] method with sparse rewards.

Usage:
------------

Run with `python main.py <options>`.

Dependencies:
------------

- [Python 3.5.2](https://www.python.org/)
- [NumPy](http://www.numpy.org/)
- [mujoco-py 1.50.1](https://github.com/openai/mujoco-py)
- [OpenAI Gym](https://gym.openai.com/)
- [PyTorch](http://pytorch.org/)
- [Plotly](https://plot.ly/python/)

Note:
------------
Obtain a 30-day free trial on the [MuJoCo website](https://www.roboti.us/license.html)
 or free license if you are a student.

Results
----------------


Acknowledgements
----------------

- [@kaixhin](https://github.com/Kaixhin) for [ACER](https://github.com/Kaixhin/ACER/)
- [@ikostrikov](https://github.com/ikostrikov) for [pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)


References
----------

* [1] [Sim-to-Real Robot Learning from Pixels with Progressive Nets](https://arxiv.org/abs/1610.04286)
* [2] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
