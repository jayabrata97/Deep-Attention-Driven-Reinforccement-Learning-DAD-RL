from aioredis import int_or_str
from env import wrapped_env
from benchmark_eval import benchmark_score
import random
import csv
from tqdm import tqdm
import numpy as np
import os
import yaml
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="path to the config file")
parser.add_argument('exp', type=int_or_str, help="experiment number")
args = parser.parse_args()
config_file = args.config
exp = args.exp

scenarios = {
        1: "../SMARTS/scenarios/sumo/intersections/1_to_1lane_left_turn_c_agents_1",
        2: "../SMARTS/scenarios/sumo/intersections/1_to_2lane_left_turn_t_agents_1",        #t-junction
        3: "../SMARTS/scenarios/sumo/augmentation_paper/left_turn",    #same as augmentation paper (with repeat true)
        # Predictive-Decision-making scenarios (4,5,6)
        4: "../SMARTS/scenarios/sumo/intersections/1_to_2lane_left_turn_c_agents_1",
        5: "../SMARTS/scenarios/sumo/merge/3lane_agents_1",
        6: "../SMARTS/scenarios/sumo/straight/3lane_overtake_agents_1",
        7: "../SMARTS/scenarios/sumo/augmentation_paper/roundabout",    #difficult one in augmentation paper (with repeat true)
        8: "../SMARTS/scenarios/sumo/augmentation_paper/roundabout_medium",     #repeat route is true
        9: "../SMARTS/scenarios/sumo/augmentation_paper/roundabout_easy",    #repeat route is true
        10: "../SMARTS/scenarios/sumo/augmentation_paper/cross" #double merge from augmentation paper (repeat route is true)
    }

with open(config_file, 'r') as f:
    cfg = yaml.safe_load(f)
reward_func = cfg['reward_func']
include_context = cfg['context']
interface_version = 4
wrapper_version = 3
max_speed = cfg['max_speed']
modify_waypoints = False    #Only using for double merge scenario

episodes = cfg['eval_eps']   #eval episodes per seed
num_seeds = cfg['num_seeds']
scene = cfg['scenario']
mode = cfg['pred_mode']

## provide absolute path
model_path = cfg['model_path']
model_name = cfg['model_name']
safety_mode = cfg['safety_mode']

remarks = cfg['remarks']
gpu = cfg['gpu']
checkpoint = cfg['checkpoint']

if mode == 'deterministic':
    determine = True
elif mode == 'sampling':
    determine = False
else:
    raise Exception("Provide a valide prediction mode: 'deterministic' or 'sampling'")

test_seeds = [random.randint(1,100) for i in range(num_seeds)]  #five random seeds to test the models on

scenario_path = scenarios[scene]

df_detailed = pd.DataFrame(columns = ["Model_Name", "Remarks", "Scenario", "Seed", "Prediction Mode", 
                             "SCORE->", "Success Rate", "Collision Rate", "Off Road Rate", "Stagnation Rate", "Comp Time Mean", "Comp Time Std",
                             "Avg Reward", "Dist2Dest", "Humanness Error", "Rule Violation", "Overall Score", 
                             "EXTRA INFO->", "Reward Function", "Context Info", "Test Episodes", "Gpu", "Checkpoint", "Path"])


sr_arr = np.zeros(num_seeds, dtype=np.float32)
coll_arr = np.zeros(num_seeds, dtype=np.float32)
off_road_arr = np.zeros(num_seeds, dtype=np.float32)
stag_arr = np.zeros(num_seeds, dtype=np.float32)

rew_arr = np.zeros(num_seeds, dtype=np.float32)
d2d_arr = np.zeros(num_seeds, dtype=np.float32)
he_arr = np.zeros(num_seeds, dtype=np.float32)
rv_arr = np.zeros(num_seeds, dtype=np.float32)
o_arr = np.zeros(num_seeds, dtype=np.float32)

for i, seed in enumerate(tqdm(test_seeds)):
    score, mean_length, mean_reward, mean_reward_components, other_metrics, comp_time_arr = benchmark_score(model_path, scenario_path, seed, reward_func, include_context, interface_version, wrapper_version, episodes, deterministic=determine, max_speed=max_speed, use_safety=safety_mode, modify_waypoints=modify_waypoints)
    
    success_rate = other_metrics[0]
    collision_rate = other_metrics[1]
    off_road_rate = other_metrics[2]
    stag_rate = other_metrics[3]
    comp_time_mean = np.mean(comp_time_arr)
    comp_time_std = np.std(comp_time_arr)
    
    d2d = score["dist_to_destination"]
    he = score["humanness_error"]
    rv = score["rule_violation"]
    overall = score["overall"]

    sr_arr[i] = success_rate
    coll_arr[i] = collision_rate
    off_road_arr[i] = off_road_rate
    stag_arr[i] = stag_rate

    rew_arr[i] = mean_reward
    d2d_arr[i] = d2d
    he_arr[i] = he
    rv_arr[i] = rv
    o_arr[i] = overall
    output = [model_name, remarks, scene, seed, mode, "-", success_rate, collision_rate, off_road_rate, stag_rate, comp_time_mean, comp_time_std,
              mean_reward, d2d, he, rv, overall, "-", reward_func, include_context, episodes, gpu, checkpoint, model_path]
    df_detailed.loc[i] = output

df_compact = pd.DataFrame(columns = ["Model Name", "Remarks", "Scenario", "Prediction Mode",
          "SCORE->", "Success Rate", "Collision Rate", "Off Road Rate", "Stagnation Rate",
          "Avg Reward", "Dist2Dest", "Humanness Error", "Rule Violation", "Overall Score", 
          "EXTRA INFO->", "Reward Function", "Context Info", "Test Episodes", "Gpu", "Checkpoint", "Path"])

avg_output = [model_name, remarks, scene, mode,
              "Mean->", np.mean(sr_arr), np.mean(coll_arr), np.mean(off_road_arr), np.mean(stag_arr),
              np.mean(rew_arr), np.mean(d2d_arr), np.mean(he_arr), np.mean(rv_arr), np.mean(o_arr),
              "-", reward_func, include_context, episodes*num_seeds, gpu, checkpoint, model_path]
std_output = [model_name, remarks, scene, mode,
              "Std->", np.std(sr_arr), np.std(coll_arr), np.std(off_road_arr), np.std(stag_arr),
              np.std(rew_arr), np.std(d2d_arr), np.std(he_arr), np.std(rv_arr), np.std(o_arr),
              "-", reward_func, include_context, episodes*num_seeds, gpu, checkpoint, model_path]

df_compact.loc[0] = avg_output
df_compact.loc[1] = std_output

test_dir = f"test_logs/new_metrics/scenario-{scene}"
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

d_file = f"{test_dir}/exp{exp}_detailed.csv"
c_file = f"{test_dir}/exp{exp}_compact.csv"

df_detailed.to_csv(d_file)
df_compact.to_csv(c_file)