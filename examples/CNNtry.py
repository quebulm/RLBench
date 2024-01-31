import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from rlbench.tasks.sorting_challenge import SortingChallenge


def preprocess_observation(observation):
    left_img = preprocess_image(observation.left_shoulder_rgb)
    right_img = preprocess_image(observation.right_shoulder_rgb)
    front_img = preprocess_image(observation.front_rgb)

    gripper_pose = torch.tensor(observation.gripper_pose, dtype=torch.float32).flatten()
    gripper_opening = torch.tensor([observation.gripper_open], dtype=torch.float32)
    joint_positions = torch.tensor(observation.joint_positions, dtype=torch.float32)
    combined_data = torch.cat((gripper_pose, gripper_opening, joint_positions), dim=0)

    # Flatten combined_data und füge eine Batch-Dimension hinzu
    combined_data = combined_data.unsqueeze(0)

    print("Left Image Shape:", left_img.shape)
    print("Right Image Shape:", right_img.shape)
    print("Front Image Shape:", front_img.shape)
    print("Combined Other Data Shape:", combined_data.shape)

    return (left_img, right_img, front_img), combined_data





# Neuer Preprocess-Ansatz für Bilder
def preprocess_image(image):
    # Normalisieren und Größe ändern (angenommen, deine Bilder sind RGB)
    return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        return self.conv_layers(x)

class CombinedModel(nn.Module):
    def __init__(self, action_shape):
        super(CombinedModel, self).__init__()
        self.cnn_encoder = CNNEncoder()

        # Angenommen, nach CNN sind die Bildfeatures von jeder Kamera 128*128
        # und du hast 3 Kameras + 7 Gripper/Joint-Daten + 1 Gripper-öffnen
        self.fc_layers = nn.Sequential(
            nn.Linear(86415, 256),
            nn.ReLU(),
            nn.Linear(256, action_shape[0])
        )

    def forward(self, images, other_data):
        img_features = [self.cnn_encoder(images[i]) for i in range(3)]

        # Stellen Sie sicher, dass other_data die gleiche Batch-Größe wie img_features hat
        if other_data.dim() == 1:
            other_data = other_data.unsqueeze(0)  # Fügt eine Batch-Dimension hinzu
        if other_data.size(0) != img_features[0].size(0):
            other_data = other_data.expand(img_features[0].size(0), -1)  # Passt die Batch-Größe an

        combined_features = torch.cat((*img_features, other_data), dim=1)
        print("Kombinierte Features Größe:", combined_features.shape)
        return self.fc_layers(combined_features)


def preprocess_observation(observation):
    left_img = preprocess_image(observation.left_shoulder_rgb)
    right_img = preprocess_image(observation.right_shoulder_rgb)
    front_img = preprocess_image(observation.front_rgb)

    gripper_pose = torch.tensor(observation.gripper_pose, dtype=torch.float32).flatten()
    gripper_opening = torch.tensor([observation.gripper_open], dtype=torch.float32)
    joint_positions = torch.tensor(observation.joint_positions, dtype=torch.float32)
    combined_data = torch.cat((gripper_pose, gripper_opening, joint_positions), dim=0)

    # Debug-Prints zur Überprüfung der Dimensionen und Strukturen
    print("Left Image Shape:", left_img.shape)
    print("Right Image Shape:", right_img.shape)
    print("Front Image Shape:", front_img.shape)
    print("Combined Other Data Shape:", combined_data.shape)

    return (left_img, right_img, front_img), combined_data

class DQNAgent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape
        self.model = CombinedModel(action_shape)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_func = nn.MSELoss()

    def train(self, images, other_data, actions, rewards, next_images, next_other_data, dones):
        # Konvertiere die Bilddaten für current und next Zustände in Tensoren, wenn sie nicht bereits Tensoren sind
        images = [torch.tensor(img, dtype=torch.float32).to(self.device) for img in images] if not isinstance(images[0],
                                                                                                              torch.Tensor) else [
            img.to(self.device) for img in images]
        next_images = [torch.tensor(img, dtype=torch.float32).to(self.device) for img in next_images] if not isinstance(
            next_images[0], torch.Tensor) else [img.to(self.device) for img in next_images]

        # send to device
        other_data = torch.tensor(other_data, dtype=torch.float32).to(self.device) if not isinstance(other_data,
                                                                                                     torch.Tensor) else other_data.to(
            self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device) if not isinstance(actions,
                                                                                               torch.Tensor) else actions.to(
            self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device) if not isinstance(rewards,
                                                                                               torch.Tensor) else rewards.to(
            self.device)
        next_other_data = torch.tensor(next_other_data, dtype=torch.float32).to(self.device) if not isinstance(
            next_other_data, torch.Tensor) else next_other_data.to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device) if not isinstance(dones,
                                                                                           torch.Tensor) else dones.to(
            self.device)

        # Forward-Pass für aktuellen und nächsten Zustand
        q_values = self.model(images, other_data)
        next_q_values = self.model(next_images, next_other_data)

        # Wähle die Q-Werte für die gewählten Aktionen
        action_indices = actions.unsqueeze(-1) # Möglicherweise muss der Shape angepasst werden
        chosen_q_values = q_values.gather(1, action_indices).squeeze()

        # Finde das Maximum der nächsten Q-Werte für alle Aktionen
        max_next_q_values = next_q_values.max(1)[0]

        # Berechne den erwarteten Q-Wert
        expected_q_values = rewards + (1 - dones) * 0.99 * max_next_q_values

        # Berechne den Verlust
        loss = self.loss_func(chosen_q_values, expected_q_values.detach())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, images, other_data):
        images = [img.to(self.device).unsqueeze(0) for img in images]
        other_data = other_data.to(self.device).view(1, -1)
        with torch.no_grad():
            action_values = self.model(images, other_data)
        # Annahme: action_values enthält 7 Werte für die Armaktionen und 1 Wert für den Greifer
        action_values_np = action_values.cpu().numpy()[0]  # Konvertiere zu Numpy-Array

        # Benutze die ersten 7 Werte direkt für die Armaktionen
        arm_actions = action_values_np[:7]
        # Entscheide über die Greiferaktion basierend auf dem achten Wert der Vorhersagen
        gripper_action = np.array([int(action_values_np[7] > 0.5)])
        # Füge die Aktionen zusammen
        action = np.hstack([arm_actions, gripper_action])

        return action


env = Environment(
    action_mode=MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    headless=True)
env.launch()

task = env.get_task(SortingChallenge)

# Reset task to get initial observation
descriptions, observation = task.reset()

# num_gripper_pose = 7      # Angenommen, basierend auf der Struktur von gripper_pose
# num_gripper_opening = 1   # Ein einzelner Wert
# num_joint_positions = 7   # Angenommen, basierend auf Ihrer Beschreibung
#
# # Gesamte Anzahl numerischer Beobachtungswerte
# total_numeric_values = num_gripper_pose + num_gripper_opening + num_joint_positions
#
# # Die Total Observation Shape für die numerischen Daten ist dann die Summe dieser Werte
# total_observation_shape_numeric = total_numeric_values  # Dies ergibt 15

action_shape = (len(observation.joint_velocities) + 1,)  # +1 für den Greifer

# Initialisiere den Agenten
agent = DQNAgent(action_shape)
training_steps = 10
episode_length = 50

# Initialzustand und -beobachtung
descriptions, obs = task.reset()
images, other_data = preprocess_observation(obs)

for i in range(training_steps):
    if i % episode_length == 0:
        if i != 0:  # Vermeide doppeltes Zurücksetzen beim ersten Durchlauf
            print('Reset Episode')
            descriptions, obs = task.reset()

        images, other_data = preprocess_observation(obs)

    action = agent.act(images, other_data)  # Act Methode erwartet jetzt direkt images und other_data
    next_obs, reward, terminate = task.step(action)
    images_next, other_data_next = preprocess_observation(next_obs)

    # Vorverarbeitung der Belohnungen und terminal Zustände für das Training, möglicherweise Anpassung für Batch-Input erforderlich
    rewards = np.array([reward])
    dones = np.array([terminate], dtype=np.float32)

    # Training des Agenten, Annahme, dass die Train Methode modifiziert wurde, um Parameter korrekt zu akzeptieren
    agent.train(images, other_data, action, rewards, images_next, other_data_next, dones)

    # Nächste Beobachtungen für den nächsten Durchgang
    images, other_data = images_next, other_data_next

print('Done with Training')
env.shutdown()

# Laden Sie das trainierte Modell
trained_model = agent.model
trained_model.eval()

# RLBench-Umgebung für SortingChallenge
sorting_env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    headless=False)
sorting_env.launch()

sorting_task = sorting_env.get_task(SortingChallenge)

# Probe-Beobachtung zu erhalten
descriptions, obs = sorting_task.reset()

episode_length = 100
for i in range(episode_length):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = sorting_task.reset()
        print(descriptions)
    images, other_data = preprocess_observation(obs)  # Entpackt die Rückgabe von preprocess_observation
    action = agent.act(images, other_data)  # Übergibt die beiden Teile separat an die act-Methode
    print(action)
    obs, reward, terminate = sorting_task.step((*action,))

print('Done')
sorting_env.shutdown()