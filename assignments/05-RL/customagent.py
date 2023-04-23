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
        """
        obs = observation
        pos_x = obs[0]
        # pos_y = obs[1]
        velocity_x = obs[2]
        velocity_y = obs[3]
        angle = obs[4]
        ang_vec = obs[5]
        action = 0

        if pos_x * velocity_x > 0 and np.abs(pos_x) > 0.015:
            if pos_x > 0:
                action = 1
            else:
                action = 3
        if angle * ang_vec > 0 and np.abs(angle) > 0.03:
            if angle > 0:
                action = 3
            else:
                action = 1
        if velocity_y < -0.25:
            action = 2

        if obs[6] and obs[7]:
            action = 0
        elif obs[6]:
            action = 3
        elif obs[7]:
            action = 1
        return action

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
