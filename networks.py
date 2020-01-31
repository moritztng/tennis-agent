import torch
import torch.nn as nn

seed = 32
torch.manual_seed(seed)

class Critic(nn.Module):
    def __init__(self, n_state, n_hidden = [500, 400], n_action=0):
        super(Critic, self).__init__()
        self.n_state = n_state
        self.n_hidden = n_hidden
        self.bn1 = nn.BatchNorm1d(n_state)
        self.fc1 = nn.Linear(n_state, n_hidden[0])
        self.fc2 = nn.Linear(n_hidden[0]+n_action, n_hidden[1])
        self.fc3 = nn.Linear(n_hidden[1], 1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.relu = nn.ReLU()
    def forward(self, state, action):
        x = state
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(torch.cat((x, action), dim=1)))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, n_state, n_action, n_hidden = [500, 400]):
        super(Actor, self).__init__()
        self.n_state = n_state
        self.n_hidden = n_hidden
        self.n_action= n_action
        self.bn1 = nn.BatchNorm1d(n_state)
        self.fc1 = nn.Linear(n_state, n_hidden[0])
        self.bn2 = nn.BatchNorm1d(n_hidden[0])
        self.fc2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.bn3 = nn.BatchNorm1d(n_hidden[1])
        self.fc3 = nn.Linear(n_hidden[1], n_action)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x
