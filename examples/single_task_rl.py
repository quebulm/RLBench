import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget


class Agent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    headless=True)
env.launch()

task = env.get_task(ReachTarget)

agent = Agent(env.action_shape)

# Laden des trainierten Modells
trained_model = agent.model.load_state_dict(torch.load('model.pth'))
trained_model.eval()
# Laden Sie das trainierte Modell
#trained_model = agent.model
#trained_model.eval()

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
