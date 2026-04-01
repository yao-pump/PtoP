# dqn_agent.py
import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class QNetwork(nn.Module):
    """
    Simple MLP for Q(s,a) approximation:
    - Input: obs_dim
    - Output: Q values of dimension n_actions
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden=(128, 128), dropout=0.0):
        super().__init__()
        h1, h2 = hidden
        self.fc1 = nn.Linear(obs_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc_out = nn.Linear(h2, n_actions)
        self.do = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1.0 / max(fan_in, 1) ** 0.5
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.do(x)
        x = F.relu(self.fc2(x))
        q = self.fc_out(x)
        return q


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        device=None,
        lr=1e-3,
        gamma=0.99,
        batch_size=128,
        target_update_interval=1000,
        replay_capacity=100_000,
        eps_start=0.20,
        eps_end=0.02,
        eps_decay_steps=50_000,
        grad_clip=5.0,
    ):
        self.device = device or torch.device("cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.grad_clip = grad_clip

        self.q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(replay_capacity)

        # epsilon schedule
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.total_steps = 0

    def epsilon(self):
        # linear decay
        t = min(self.total_steps, self.eps_decay_steps)
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - t / self.eps_decay_steps)

    @torch.no_grad()
    def select_action(self, obs_np: np.ndarray):
        """
        obs_np: numpy vector of shape (obs_dim,)
        Returns: action index as int
        """
        self.total_steps += 1
        if random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        x = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q_net(x)
        a = int(torch.argmax(q, dim=-1).item())
        return a

    def push(self, s, a, r, s2, done: bool):
        s_t = torch.tensor(s, dtype=torch.float32)
        s2_t = torch.tensor(s2, dtype=torch.float32)
        r_t = torch.tensor([r], dtype=torch.float32)
        a_t = torch.tensor([a], dtype=torch.int64)
        d_t = torch.tensor([done], dtype=torch.bool)
        self.replay.push(s_t, a_t, r_t, s2_t, d_t)

    def optimize(self):
        if len(self.replay) < max(self.batch_size, 2048):  # warm-up first
            return None

        batch = self.replay.sample(self.batch_size)

        state_b = torch.stack(batch.state).to(self.device)          # [B, obs]
        action_b = torch.cat(batch.action).to(self.device)          # [B]
        reward_b = torch.cat(batch.reward).to(self.device)          # [B]
        next_state_b = torch.stack(batch.next_state).to(self.device) # [B, obs]
        done_b = torch.cat(batch.done).to(self.device)              # [B] bool

        q = self.q_net(state_b)                                     # [B, A]
        q_sa = q.gather(1, action_b.view(-1, 1)).squeeze(1)         # [B]

        with torch.no_grad():
            # Double DQN: select action with online network, estimate value with target network
            next_q_online = self.q_net(next_state_b)                # [B, A]
            next_a = torch.argmax(next_q_online, dim=1, keepdim=True)  # [B,1]
            next_q_target = self.target_net(next_state_b)           # [B, A]
            next_q = next_q_target.gather(1, next_a).squeeze(1)     # [B]
            target = reward_b + (~done_b).float() * self.gamma * next_q

        loss = F.smooth_l1_loss(q_sa, target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optim.step()

        # update target network
        if (self.total_steps % self.target_update_interval) == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def save(self, path="dqn.pt"):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path="dqn.pt"):
        sd = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(sd)
        self.target_net.load_state_dict(self.q_net.state_dict())
