# This is demo script to visualtion the simulation of a trained agaent.
# Specify the model_path for the agent you want to use.
# This example assumes the agent is trained using PPO, the code can be modified if you want to use SAC or other models instead.

from stable_baselines3 import PPO
from env import wrapped_env

scenario_path = "../SMARTS/scenarios/sumo/merge/3lane_agents_1"

env = wrapped_env(scenario_path, "dense", True, interface_version=4, headless=True)

model_path = "-"
model = PPO.load(model_path)

obs = env.reset()
for i in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()

env.close()
del model
