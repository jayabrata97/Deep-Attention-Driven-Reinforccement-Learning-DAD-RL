import gymnasium as gym
from stable_baselines3 import PPO
from smarts.core.agent_interface import AgentInterface, NeighborhoodVehicles, DrivableAreaGridMap, OGM, DoneCriteria
from smarts.core.controllers import ActionSpaceType
from env import wrapped_env

# scenario_path = "../SMARTS/scenarios/sumo/straight/3lane_overtake_agents_1"
scenario_path = "../SMARTS/scenarios/sumo/merge/3lane_agents_1"
# scenario_path = "../SMARTS/scenarios/sumo/intersections/1_to_2lane_left_turn_c_agents_1"

env = wrapped_env(scenario_path, 'type3', True, headless=True) 

model = PPO.load("../attn_model_with_sb3/logs_experimental/Attention_PPO_frozen/checkpoints/Attention_PPO_frozen_2162688_steps.zip", device='cuda:1')
# print(f"\n\n Attention PPO Model Policy:\n{model.policy} \n\n")

obs = env.reset()
for i in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()

env.close()
del model