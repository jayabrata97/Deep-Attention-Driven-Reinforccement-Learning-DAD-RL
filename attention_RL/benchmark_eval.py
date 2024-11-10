#TO DO: CHECK WHICH benchmark_score is used
import numpy as np
from typing import Dict
import pprint
import stable_baselines3 as sb3
from sympy import comp
from torch.utils.tensorboard import SummaryWriter
import random

from smarts.env.gymnasium.wrappers.metric.types import Record
from smarts.env.gymnasium.wrappers.metric.utils import multiply, op_dataclass
from smarts.env.gymnasium.wrappers.metric.formula import FormulaBase, Formula

from env import benchmark_env

def safety_layer(obs, action, max_speed):
    #dummy function
    #can be designed to optimize actions by the agent to ensure extra safety
    return action

def benchmark_score(model_path:str, scenario_path:str, seed, reward_func, include_context, interface_version, wrapper_version, episodes=5, deterministic=False, device='cuda:0', max_speed=15, use_safety=False, modify_waypoints=False):

    env = benchmark_env(scenario_path, reward_func, include_context, interface_version, wrapper_version, seed, True, max_speed=max_speed)
    if scenario_path == "../SMARTS/scenarios/sumo/augmentation_paper/cross":
        modify_waypoints = True
    else:
        modify_waypoints = False
        
    env.waypoints_modifier(modify_waypoints)

    if 'PPO' in model_path:
        trained_agent = sb3.PPO.load(model_path, env, device=device)
    elif 'SAC' in model_path:
        trained_agent = sb3.SAC.load(model_path, env, device=device)
    env.seed(seed)  #to avoid model seed overwriting env seed
    obs = env.reset()
    num_resets= 0 

    ###changes to collect avg reward and episode length
    ep_length = 0
    ep_reward = 0
    mean_length = 0
    mean_reward = 0

    ep_mean_speed = 0
    ep_progress = 0
    ep_off_route = 0
    ep_wrong_way = 0
    ep_not_moving = 0
    mean_speed = 0
    mean_progress = 0
    mean_goal = 0
    mean_crash = 0
    mean_off_road = 0
    mean_off_route = 0
    mean_wrong_way = 0
    mean_not_moving = 0

    mean_stag = 0
    comp_time_arr = np.zeros(episodes, dtype=np.float32)

    i=0

    while num_resets < episodes:
        i+=1

        action, _ = trained_agent.predict(obs, deterministic=deterministic)

        if use_safety:
            # print("\n ------- Using Safety Layer ------- \n")
            action = safety_layer(obs, action, max_speed)
        else: pass

        obs, reward, done, info = env.step(action)

        ep_length += 1
        ep_reward += reward

        ######### NEW FEATURE ########
        progress = info['RewardComponents']['progress']
        ep_goal = info['RewardComponents']['goal']
        ep_crash = info['RewardComponents']['crash']
        ep_off_road = info['RewardComponents']['off_road']
        off_route = info['RewardComponents']['off_route']
        wrong_way = info['RewardComponents']['wrong_way']
        not_moving = info['RewardComponents']['not_moving']

        ep_stag = abs(info["is_stagnation"])

        ep_progress += progress
        ep_mean_speed = ep_progress/(i*0.1)
        ep_off_route += off_route
        ep_wrong_way += wrong_way
        ep_not_moving += not_moving

        if done:
            comp_time_arr[num_resets] = ep_length

            num_resets += 1
            obs = env.reset()

            mean_length = (mean_length*(num_resets-1) + ep_length)/num_resets
            mean_reward = (mean_reward*(num_resets-1) + ep_reward)/num_resets
            ep_length = 0
            ep_reward = 0

            mean_progress = (mean_progress*(num_resets-1) + ep_progress)/num_resets
            mean_speed = (mean_speed*(num_resets-1) + ep_mean_speed)/num_resets
            mean_goal = (mean_goal*(num_resets-1) + ep_goal)/num_resets
            mean_crash = (mean_crash*(num_resets-1) + ep_crash)/num_resets
            mean_off_road = (mean_off_road*(num_resets-1) + ep_off_road)/num_resets
            mean_off_route = (mean_off_route*(num_resets-1) + ep_off_route/i)/num_resets        #normalized frequency of off_route in an episode(0-1)
            mean_wrong_way = (mean_wrong_way*(num_resets-1) + ep_wrong_way/i)/num_resets        #normalized frequency of wrong_way in an episode(0-1)
            mean_not_moving = (mean_not_moving*(num_resets-1) + ep_not_moving/i)/num_resets     #normalized frequency of not_moving in an episode(0-1)
            
            mean_stag = (mean_stag*(num_resets-1) + ep_stag)/num_resets

            ep_mean_speed = 0
            ep_progress = 0
            ep_off_route = 0
            ep_wrong_way = 0
            ep_not_moving = 0
            i=0

    mean_reward_components = np.array([mean_progress, mean_speed, mean_goal, mean_crash, mean_off_road, mean_off_route, mean_wrong_way, mean_not_moving])

    other_metrics = np.array([mean_goal, mean_crash, mean_off_road, mean_stag])

    records = env.records()

    records_cumulative: Dict[str, Dict[str, Record]] = {}
    records_cumulative.update(records)

    records_sum = {}
    for scen, agents in records_cumulative.items():
        records_sum[scen] = {}
        for agent, data in agents.items():
            records_sum[scen][agent] = Record(
                costs=op_dataclass(data.costs, data.counts.episodes, multiply),
                counts=data.counts,
            )

    formula: FormulaBase = Formula()
    score = formula.score(records_sum=records_sum)

    print("\nSCORE")
    pprint.pprint(score)

    env.close()
    trained_agent.env.close()
    del trained_agent
    
    return score, mean_length, mean_reward, mean_reward_components, other_metrics, comp_time_arr

def benchmark_logger(log_dir, scenarios, scenes, model_path, SEED, N_eval_envs, reward_func, include_context, interface_version, wrapper_version, eps, i, TIMESTEPS, max_speed, modify_waypoints=False):

    if SEED == "mix":
        eval_seeds = [random.randint(1,100) for i in range(N_eval_envs)]
    elif SEED == 'default':
        eval_seeds = [None]*N_eval_envs
    elif isinstance(SEED, int):
        eval_seeds = [SEED]*N_eval_envs
    else: raise Exception("Specify seed as 'default', 'mix', or an int")

    print("\n Evaluation seeds: ", eval_seeds)

    if N_eval_envs%len(scenes) == 0:
        multiplier = N_eval_envs//len(scenes)
        eval_scenes = scenes*multiplier
        eval_scenarios = []
        for s in eval_scenes:
            eval_scenarios.append(scenarios[s])
    else:
        raise Exception("Number of envs must be a mutiple of number of training scenarios")

    total_score_overall = 0
    total_score_d2d = 0
    total_score_he = 0
    total_score_rv = 0
    total_score_time = 0
    total_length = 0
    total_reward = 0
    total_rew_components = np.zeros(8)

    Score = [{}]*N_eval_envs
    Mean_length = np.zeros(N_eval_envs)
    Mean_reward = np.zeros(N_eval_envs)
    Rew_components = np.zeros((N_eval_envs,8))

    for j, (scenario, eval_seed) in enumerate(zip(eval_scenarios, eval_seeds)):
        Score[j], Mean_length[j], Mean_reward[j], Rew_components[j], _, _ = benchmark_score(model_path, scenario, eval_seed, reward_func, include_context, interface_version, wrapper_version, episodes=eps, max_speed=max_speed, modify_waypoints=modify_waypoints)

    writer = SummaryWriter(log_dir=f"{log_dir}/tensorboard/benchmark")
    #separate loop for writing to tensorboard in case of intermediate crashing
    for j,(score,mean_length,mean_reward,rew_components) in enumerate(zip(Score,Mean_length,Mean_reward,Rew_components)):
        total_score_overall += score["overall"]
        total_score_d2d += score["dist_to_destination"]
        total_score_he += score["humanness_error"]
        total_score_rv += score["rule_violation"]
        total_score_time += score["time"]
        total_length += mean_length
        total_reward += mean_reward
        total_rew_components += rew_components

    writer.add_scalar(f"Benchmark_Combined/Overall Score", total_score_overall/N_eval_envs, i*TIMESTEPS)
    writer.add_scalar(f"Benchmark_Combined/Distance to destination", total_score_d2d/N_eval_envs, i*TIMESTEPS)
    writer.add_scalar(f"Benchmark_Combined/Humanness error", total_score_he/N_eval_envs, i*TIMESTEPS)
    writer.add_scalar(f"Benchmark_Combined/Rule Violation", total_score_rv/N_eval_envs, i*TIMESTEPS)
    writer.add_scalar(f"Benchmark_Combined/Time", total_score_time/N_eval_envs, i*TIMESTEPS)
    writer.add_scalar(f"Eval_Combined/Mean_ep_length", total_length/N_eval_envs, i*TIMESTEPS)
    writer.add_scalar(f"Eval_Combined/Mean_reward", total_reward/N_eval_envs, i*TIMESTEPS)
    writer.add_scalar(f"Eval_Combined/Mean_progress", total_rew_components[0]/N_eval_envs, i*TIMESTEPS)
    writer.add_scalar(f"Eval_Combined/Mean_speed", total_rew_components[1]/N_eval_envs, i*TIMESTEPS)
    writer.add_scalar(f"Eval_Combined/Mean_goal", total_rew_components[2]/N_eval_envs, i*TIMESTEPS)
    writer.add_scalar(f"Eval_Combined/Mean_crash", total_rew_components[3]/N_eval_envs, i*TIMESTEPS)
    writer.add_scalar(f"Eval_Combined/Mean_off_road", total_rew_components[4]/N_eval_envs, i*TIMESTEPS)
    writer.add_scalar(f"Eval_Combined/Mean_off_route", total_rew_components[5]/N_eval_envs, i*TIMESTEPS)
    writer.add_scalar(f"Eval_Combined/Mean_wrong_way", total_rew_components[6]/N_eval_envs, i*TIMESTEPS)
    writer.add_scalar(f"Eval_Combined/Mean_not_moving", total_rew_components[7]/N_eval_envs, i*TIMESTEPS)
    writer.close()