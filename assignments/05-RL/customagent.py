import gymnasium as gym
import numpy as np


class Agent:

    """
    A maybe not reinforcement learning agent that learns to play a Lunar Lander game.
    https://gymnasium.farama.org/environments/box2d/lunar_lander/
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        """
        init variables
        """
        self.action_space = action_space
        self.observation_space = observation_space

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Take an observation and return an action.
        Args:
            observation (gym.spaces.Box): _description_
        Returns:
            gym.spaces.Discrete: _description_
        from https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
        """
        thres = 0.45
        angle_targ = (
            observation[0] * 0.5 + observation[2] * 1.0
        )  # angle should point towards center
        if angle_targ > thres:
            angle_targ = thres  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -thres:
            angle_targ = -thres
        hover_targ = 0.55 * np.abs(
            observation[0]
        )  # target y should be proportional to horizontal offset

        angle_todo = (angle_targ - observation[4]) * 0.5 - (observation[5]) * 1.0
        hover_todo = (hover_targ - observation[1]) * 0.5 - (observation[3]) * 0.5

        if observation[6] or observation[7]:  # legs have contact
            angle_todo = 0
            hover_todo = (
                -(observation[3]) * 0.5
            )  # override to reduce fall speed, that's all we need after contact

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
        return a

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
         Take an observation, a reward, a boolean indicating whether the episode has
         terminated, and a boolean indicating whether the episode was truncated
        Args:
            observation (gym.spaces.Box): _description_
            reward (float): _description_
            terminated (bool): _description_
            truncated (bool): _description_
        """

        pass
