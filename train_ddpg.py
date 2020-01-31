import torch, random, argparse
import torch.nn as nn
from tqdm import tqdm_notebook
from visdom import Visdom
from unityagents import UnityEnvironment
import numpy as np
from networks import Actor, Critic
from replay_buffer import ReplayMemory

seed = 32
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Solve Environment with MADDPG')
parser.add_argument('--environment', default='./Tennis_Linux/Tennis.x86_64')
parser.add_argument('--replay_capacity', type=int, default=1e6)
parser.add_argument('--lr_actor', type=float, default=1e-4)
parser.add_argument('--lr_critic', type=float, default=1e-4)
parser.add_argument('--weight_decay_critic', type=float, default=0)
parser.add_argument('--discount_factor', type=float, default=0.9)
parser.add_argument('--tau', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--noise_std_start', type=float, default=0.3)
parser.add_argument('--noise_std_decay', type=float, default=0.999)
parser.add_argument('--noise_std_min', type=float, default=0.01)
parser.add_argument('--episodes', type=int, default=10000)
parser.add_argument('--max_t', type=int, default=int(1e9))
parser.add_argument('--n_steps', type=int, default=20)
parser.add_argument('--iterations', type=int, default=10)
parser.add_argument('--print_episodes', type=int, default=200)

args = parser.parse_args()

env = UnityEnvironment(file_name=args.environment)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

vis = Visdom()
win_score = None
win_actor_score = None
win_critic_loss = None

actor = Actor(state_size*2, action_size*2).to(device)
actor_target = Actor(state_size*2, action_size*2).to(device)
critic = Critic(state_size*2, n_action = action_size*2).to(device)
critic_target = Critic(state_size*2, n_action = action_size*2).to(device)
for target_param, param in zip(critic_target.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(actor_target.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)
replay_buffer = ReplayMemory(args.replay_capacity)
criterion = nn.MSELoss()
optim_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic, weight_decay=args.weight_decay_critic)
optim_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)

loss_critic = []
score_actor = []
score = 0
steps = 0
noise_std = args.noise_std_start

for i in range(args.episodes):
    env_info = env.reset(train_mode=True)[brain_name]
    state = torch.from_numpy(env_info.vector_observations).view(-1).float().to(device)
    for t in range(args.max_t):
        with torch.no_grad():
            actor.eval()
            action = torch.clamp(actor_target(state.unsqueeze(0))+torch.zeros((1, action_size*2)).normal_(0,noise_std).to(device),-1,1).squeeze().float()#+ ou_process.sample()
            actor.train()
            env_info = env.step(torch.stack((action[:action_size], action[action_size:])).to('cpu').numpy())[brain_name]
            next_state = torch.from_numpy(env_info.vector_observations).view(-1).float()
            reward = torch.tensor(env_info.rewards).sum().float()
            score += reward.item()
            done = torch.tensor(env_info.local_done[0] or env_info.local_done[1]).float()
            replay_buffer.push(state.to('cpu'), action.to('cpu'), next_state, reward, done)
        if((steps+1)%args.n_steps==0 and len(replay_buffer)>=args.batch_size):
            for iteration in range(args.iterations):
                with torch.no_grad():
                    sample = replay_buffer.sample(args.batch_size)
                    states = torch.stack([row.state for row in sample]).to(device)
                    actions = torch.stack([row.action for row in sample]).to(device)
                    rewards = torch.stack([row.reward for row in sample]).unsqueeze(1).to(device)
                    next_states = torch.stack([row.next_state for row in sample]).to(device)
                    dones = torch.stack([row.done for row in sample]).unsqueeze(1).to(device)
                    targets = rewards + (1-dones) * args.discount_factor * critic_target(next_states, actor_target(next_states))
                optim_critic.zero_grad()
                predictions = critic(states, actions)
                loss = criterion(predictions, targets)
                loss_critic.append(loss.item())
                loss.backward()
                optim_critic.step()
                optim_actor.zero_grad()
                loss = -critic(states, actor(states)).mean()
                score_actor.append(-loss.item())
                loss.backward()
                optim_actor.step()
                with torch.no_grad():
                    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - args.tau) + param.data * args.tau)
                    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - args.tau) + param.data * args.tau)
        steps += 1
        if done.item():
            break
        state = next_state.to(device)
    noise_std = max(noise_std*args.noise_std_decay,args.noise_std_min)
    if (i+1)%args.print_episodes==0:
        avg_score = score/args.print_episodes
        avg_score_actor = torch.tensor(score_actor).mean().item()
        avg_loss_critic = torch.tensor(loss_critic).mean().item()
        win_score = vis.line(X=[i], Y=[avg_score], win = win_score, update="append") if win_score else vis.line(X=[i], Y=[avg_score], opts=dict(title="Average Score", xlabel="Episodes"))
        win_actor_score = vis.line(X=[i], Y=[avg_score_actor], win = win_actor_score, update="append") if win_actor_score else vis.line(X=[i], Y=[avg_score_actor], opts=dict(title="Average Actor Score", xlabel="Episodes"))
        win_critic_loss = vis.line(X=[i], Y=[avg_loss_critic], win = win_critic_loss, update="append") if win_critic_loss else vis.line(X=[i], Y=[avg_loss_critic], opts=dict(title="Average Critic Loss", xlabel="Episodes"))
        print("Episode: {}/{}, Score: {:.4f}, Score Actor: {:.4f}, Loss Critic: {:.7f}, Noise Std: {:.4f}".format(i+1, args.episodes, avg_score, avg_score_actor, avg_loss_critic, noise_std))

        score_actor = []
        loss_critic = []
        score = 0

env.close()
torch.save(actor.state_dict(),"checkpoint_ddpg.pth")
