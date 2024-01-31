import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import ReachTarget
from rlbench.tasks.sorting_challenge import SortingChallenge

def preprocess_observation(observation):
    # Kamerabilder Flach
    left_img = observation.left_shoulder_rgb.flatten()
    right_img = observation.right_shoulder_rgb.flatten()

    # Beobachtungen in einem  rray
    combined_obs = np.concatenate([left_img, right_img, observation.joint_positions], axis=0)
    return combined_obs


# Definieren des DQN-Agenten
class DQNAgent(object):
    def __init__(self, action_shape, observation_shape):
        self.action_shape = action_shape
        self.model = nn.Sequential(
            nn.Linear(np.prod(observation_shape), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # action_shape ist jetzt 8
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        print(torch.cuda.is_available())
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(torch.cuda.current_device()))
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).view(1, -1)
        if torch.cuda.is_available():
            obs = obs.cuda()
        with torch.no_grad():
            action_values = self.model(obs)
        # sicherzustellen dass  ein Array für den Greiferwert übergeben wird
        return np.hstack([action_values.cpu().numpy()[0][:7], np.array([int(action_values.cpu().numpy()[0][7] > 0.5)])])

    def train(self, state, action, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        state = state.cuda()
        next_state = next_state.cuda()
        action = action.cuda()
        reward = reward.cuda()
        done = done.cuda()

        q_values = self.model(state)
        next_q_values = self.model(next_state)
        q_value = q_values.gather(0, action.argmax())
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + 0.99 * next_q_value * (1 - done)

        loss = nn.MSELoss()(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# RLBench-Umgebung
env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    headless=True)
env.launch()

task = env.get_task(ReachTarget)

# Probe-Beobachtung zu erhalten
descriptions, observation = task.reset()


# Obervation
left_shoulder_image_shape = observation.left_shoulder_rgb.shape
right_shoulder_image_shape = observation.right_shoulder_rgb.shape
joint_positions_shape = observation.joint_positions.shape

total_observation_shape = (np.prod(left_shoulder_image_shape) +
                           np.prod(right_shoulder_image_shape) +
                           np.prod(joint_positions_shape), )

# TODO Action Shape bestimmen (muss angepasst werden)
action_shape = (len(observation.joint_velocities) + 1, )  # +1 für den Greifer

# Initialisiere den Agenten
agent = DQNAgent(action_shape, total_observation_shape)

training_steps = 120
episode_length = 40
obs = None
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        print(descriptions)
    processed_obs = preprocess_observation(obs)
    action = agent.act(processed_obs)
    print(action)
    next_obs, reward, terminate = task.step((*action,))
    processed_obs = preprocess_observation(obs)
    processed_next_obs = preprocess_observation(next_obs)
    agent.train(processed_obs, action, processed_next_obs, reward, terminate)
    obs = next_obs

print('Done')
env.shutdown()
