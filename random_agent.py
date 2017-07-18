import argparse
import numpy as np
from jaco_arm import JacoEnv
import mujoco_py

import matplotlib.pyplot as plt
plt.ion()

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--width', type=int, default=64, help='RGB width')
parser.add_argument('--height', type=int, default=64, help='RGB height')
parser.add_argument('--frame_skip', type=int, default=100, help="Frame skipping in environment. Repeats last agent action.")
parser.add_argument('--rewarding_distance', type=float, default=0.1, help='Distance from target at which reward is provided.')
parser.add_argument('--control_magnitude', type=float, default=0.8, help='Fraction of actuator range used as control inputs.')
parser.add_argument('--reward_continuous', action='store_true', help='if True, provides rewards at every timestep')
parser.add_argument('--render', action='store_true', help='if True, sets up MuJoCo Viewer instead of Matplotlib')


class JacoEnvRandomAgent():
    def __init__(self, width, height, frame_skip, rewarding_distance, control_magnitude,
                 reward_continuous, render):
        self.env = JacoEnv(width, height, frame_skip, rewarding_distance,
                           control_magnitude, reward_continuous)
        self.render = render

    def run(self):
        (_, _, obs_rgb_view2) = self.env.reset()

        if self.render:
            viewer = mujoco_py.MjViewer(self.env.sim)
        else:
            f, ax = plt.subplots()
            im = ax.imshow(obs_rgb_view2)

        while True:
            self.env.reset()

            while True:

                # random action selection
                action = np.random.choice([0, 1, 2, 3, 4], 6)

                # take the random action and observe the reward and next state (2 rgb views and proprioception)
                (obs_joint, obs_rgb_view1, obs_rgb_view2), reward, done = self.env.step(action)

                # print("action : ", action)
                # print("reward : ", reward)

                if done:
                    break

                if self.render:
                    viewer.render()
                else:
                    im.set_data(obs_rgb_view2)
                    plt.draw()
                    plt.pause(0.1)


if __name__ == '__main__':
    args = parser.parse_args()
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    agent = JacoEnvRandomAgent(args.width, args.height, args.frame_skip,
                               args.rewarding_distance, args.control_magnitude,
                               args.reward_continuous, args.render)
    agent.run()
