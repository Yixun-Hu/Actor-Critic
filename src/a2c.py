import torch
import torch.nn as nn
from torch.distributions import Categorical


class A2C(nn.Module):

    def __init__(self, env, hidden_size=128, gamma=.99, random_seed=None):
        """
        Assumes fixed continuous observation space
        and fixed discrete action space (for now)

        :param env: target gym environment
        :param gamma: the discount factor parameter for expected reward function :float
        :param random_seed: random seed for experiment reproducibility :float, int, str
        """
        super().__init__()

        if random_seed:
            env.seed(random_seed)
            torch.manual_seed(random_seed)

        self.env = env
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.in_size = len(env.observation_space.sample().flatten())
        self.out_size = self.env.action_space.n

        self.actor = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.out_size)
        ).double()

        self.critic = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).double()

    def train_env_episode(self, render=False):
        """
        Runs one episode and collects critic values, expected return,
        :return: A tensor with total/expected reward, critic eval, and action information
        """
        rewards = []
        critic_vals = []
        action_lp_vals = []

        # Run episode and save information

        observation = self.env.reset()
        done = False
        while not done:
            if render:
                self.env.render()

            observation = torch.from_numpy(observation).double()

            # Get action from actor
            action_logits = self.actor(observation)
            # print(f"action_logits: {action_logits}")
            # action_logits: tensor([0.0769, 0.3220], dtype=torch.float64, grad_fn=<ViewBackward0>)

            dist = Categorical(logits=action_logits)
            action = dist.sample()
            # print(f"action: {action}")
            # action: 1

            # Get action probability
            action_log_prob = dist.log_prob(action)
            # print(f"action_log_prob: {action_log_prob}")
            # action_log_prob: -0.5780688090522006

            # Get value from critic
            pred = torch.squeeze(self.critic(observation).view(-1))

            # Write prediction and action/probabilities to arrays
            action_lp_vals.append(action_log_prob)
            critic_vals.append(pred)

            # Send action to environment and get rewards, next state

            observation, reward, done, info = self.env.step(action.item())
            rewards.append(torch.tensor(reward).double())

        total_reward = sum(rewards)

        # Convert reward array to expected return and standardize
        for t_i in range(len(rewards)):
            G = 0
            for t in range(t_i, len(rewards)):
                G += rewards[t] * (self.gamma ** (t - t_i))
            rewards[t_i] = G

        # Convert output arrays to tensors using torch.stack
        def f(inp):
            return torch.stack(tuple(inp), 0)

        # Standardize rewards
        rewards = f(rewards)
        rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards) + .000000000001)

        return rewards, f(critic_vals), f(action_lp_vals), total_reward

    def test_env_episode(self, render=True):
        """
        Run an episode of the environment in test mode
        :param render: Toggle rendering of environment :bool
        :return: Total reward :int
        """
        observation = self.env.reset()
        rewards = []
        done = False
        while not done:

            if render:
                self.env.render()

            observation = torch.from_numpy(observation).double()

            # Get action from actor
            action_logits = self.actor(observation)
            action = Categorical(logits=action_logits).sample()

            observation, reward, done, info = self.env.step(action.item())
            rewards.append(reward)

        return sum(rewards)

    @staticmethod
    def compute_loss(
        action_p_vals: torch.Tensor,
        G: torch.Tensor,
        V: torch.Tensor,
        critic_loss: nn.Module | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute actor and critic losses for the A2C update.

        Parameters
        ----------
        action_p_vals : torch.Tensor
            Log probabilities of sampled actions at each timestep.
            Shape should match ``G`` and ``V`` (typically ``[T]``).
        G : torch.Tensor
            Discounted returns for each timestep (target values for critic).
            Shape should match ``action_p_vals`` and ``V``.
        V : torch.Tensor
            Critic-predicted state values for each timestep.
            Shape should match ``action_p_vals`` and ``G``.
        critic_loss : nn.Module | None, optional
            Critic loss function module. If ``None``, ``nn.SmoothL1Loss()``
            is used.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple ``(actor_loss, critic_loss_value)`` where both are scalar
            tensors used for backpropagation.

        Raises
        ------
        AssertionError
            Raised when ``action_p_vals``, ``G``, and ``V`` do not share the
            same shape.
        """
        if critic_loss is None:
            critic_loss = nn.SmoothL1Loss()

        assert action_p_vals.shape == G.shape == V.shape

        # Detach V so actor loss does not backpropagate into critic network.
        advantage = G - V.detach()
        actor_loss = -(torch.sum(action_p_vals * advantage))
        critic_loss_value = critic_loss(V, G)
        return actor_loss, critic_loss_value
