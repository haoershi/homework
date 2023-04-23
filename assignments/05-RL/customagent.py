import gymnasium as gym
import numpy as np


class Agent:
    """
    Agent Class
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
        Takes an observation and returns an action.
        from https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py 
        """
        angle_targ = observation[0] * 0.5 + observation[2] * 1.0  # angle should point towards center
        if angle_targ > 0.4:
            angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4:
            angle_targ = -0.4
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
        Takes an observation, a reward, a boolean indicating whether the episode has terminated,
        and a boolean indicating whether the episode was truncated.
        """

        pass
