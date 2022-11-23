import sys
import subprocess
#Installing preqs
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gym_super_mario_bros==7.3.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gym_super_mario_bros==7.3.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'stable-baselines3[extra]'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--pre', 'torch', '--extra-index-url', 'https://download.pytorch.org/whl/nightly/cpu'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--pre', 'torchvision', '--extra-index-url', 'https://download.pytorch.org/whl/nightly/cpu'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--pre', 'torchaudio', '--extra-index-url', 'https://download.pytorch.org/whl/nightly/cpu'])


# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt
# Import os for file path management
import os 
# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback


print("Initializing...")
# Setup game
print("Setting up game...")
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

"""
# Create a flag - restart or not
print("Verifying game loads...")
done = True
# Loop through each frame in the game
for step in range(1000): 
    # Start the game to begin with 
    if done: 
        # Start the gamee
        env.reset()
    # Do random actions
    state, reward, done, info = env.step(env.action_space.sample())
    # Show the game on the screen
    env.render()
# Close the game
env.close()
"""
print("Remember to close 'Figure 1' Window to continue")



print("Initializing environment...")
# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify the controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')

state = env.reset()

state, reward, done, info = env.step([5])

plt.figure(figsize=(20,16))
for idx in range(state.shape[3]):
    plt.subplot(1,4,idx+1)
    plt.imshow(state[0][:,:,idx])
plt.show()


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

print("Creating checkpoint...")
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

# This is the AI model started
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, 
            n_steps=512) 

# Train the AI model, this is where the AI model starts to learn
print("Training started. Please wait. This will take a while...")
model.learn(total_timesteps=1000000, callback=callback)

model.save('thisisatestmodel')
print("Training Complete.")


# Load model
model = PPO.load('./train/best_model_1000000')

state = env.reset()

# Start the game 
state = env.reset()
# Loop through the game
while True: 
    
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
