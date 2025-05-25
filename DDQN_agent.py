import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, num_clients, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_clients, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_clients)
        )

    def forward(self, x):
        return self.net(x)

class DDQNAgent:
    def __init__(self, num_clients, lr=1e-3, gamma=0.99, buffer_size=5000):
        self.num_clients = num_clients
        self.gamma = gamma

        self.policy_net = QNetwork(num_clients)
        self.target_net = QNetwork(num_clients)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.opt = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    def select_action(self, state, eps, k):
        if random.random() < eps:
            return torch.randint(
                low=0,
                high=self.num_clients,
                size=(k,),
                dtype=torch.long,
                device=state.device
            )
        with torch.no_grad():
            qvals = self.policy_net(state).squeeze(0)  # [N]
            _, idx = qvals.topk(k)  # idx: [k]
            return idx.long()  # [k]

    def push_transition(self, state, action, reward, next_state):
        self.buffer.push(state, action, reward, next_state)

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        transitions = self.buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        S = torch.stack(batch.state)
        A = torch.stack(batch.action)
        R = torch.tensor(batch.reward,device=S.device)
        S2 = torch.stack(batch.next_state)

        Qs = self.policy_net(S)

        A = A.to(device=Qs.device, dtype=torch.long)
        if A.dim() == 1:
            A = A.unsqueeze(1)

        Q_s_a = Qs.gather(1, A)

        with torch.no_grad():
            next_q = self.policy_net(S2)
            _, best_idx = next_q.max(dim=1)
            Q2 = self.target_net(S2)
            Q2_s2_best = Q2[torch.arange(batch_size), best_idx]
            target = R + self.gamma * Q2_s2_best

        loss = ((Q_s_a.mean(dim=1) - target) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.steps += 1
        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

