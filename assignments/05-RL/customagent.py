import gymnasium as gym


class Agent:
    """_summary_"""

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """_summary_

        Args:
            observation (gym.spaces.Box): _description_

        Returns:
            gym.spaces.Discrete: _description_
        """
        return self.action_space.sample()

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """_summary_

        Args:
            observation (gym.spaces.Box): _description_
            reward (float): _description_
            terminated (bool): _description_
            truncated (bool): _description_
        """
        pass
