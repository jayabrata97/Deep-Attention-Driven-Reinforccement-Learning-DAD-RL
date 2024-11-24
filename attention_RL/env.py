from smarts.core.agent_interface import (
    AgentInterface,
    NeighborhoodVehicles,
    DrivableAreaGridMap,
    DoneCriteria,
)
from smarts.core.controllers import ActionSpaceType
from smarts.env.gymnasium.wrappers.api_reversion import Api021Reversion
from smarts.env.gymnasium.wrappers.single_agent import SingleAgent
from smarts.env.gymnasium.wrappers.metric.metrics import Metrics
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym

from wrappers import ObsWrapper, StateWrapper

done_criteria = DoneCriteria(
    collision=True,
    off_road=True,
    off_route=False,
    on_shoulder=False,
    wrong_way=False,
    not_moving=False,
    agents_alive=None,
)

agent_interface = AgentInterface(
    max_episode_steps=800,
    neighborhood_vehicle_states=NeighborhoodVehicles(radius=60),
    drivable_area_grid_map=DrivableAreaGridMap(256, 256, 32 / 256),
    top_down_rgb=False,
    lidar_point_cloud=False,
    action=ActionSpaceType.Continuous,
    waypoint_paths=True,
    done_criteria=done_criteria,
)

agent_interface_v2 = AgentInterface(
    max_episode_steps=800,
    neighborhood_vehicle_states=NeighborhoodVehicles(radius=30),
    drivable_area_grid_map=DrivableAreaGridMap(128, 128, 50 / 128),
    top_down_rgb=False,
    lidar_point_cloud=False,
    action=ActionSpaceType.Continuous,
    waypoint_paths=True,
    done_criteria=done_criteria,
)

agent_interface_v3 = AgentInterface(
    max_episode_steps=800,
    neighborhood_vehicle_states=NeighborhoodVehicles(radius=30),
    drivable_area_grid_map=DrivableAreaGridMap(128, 128, 50 / 128),
    top_down_rgb=False,
    lidar_point_cloud=False,
    action=ActionSpaceType.Lane,
    waypoint_paths=True,
    done_criteria=done_criteria,
)

agent_interface_v4 = AgentInterface(
    max_episode_steps=800,
    neighborhood_vehicle_states=NeighborhoodVehicles(radius=30),
    drivable_area_grid_map=DrivableAreaGridMap(128, 128, 50 / 128),
    top_down_rgb=False,
    lidar_point_cloud=False,
    action=ActionSpaceType.LaneWithContinuousSpeed,
    waypoint_paths=True,
    done_criteria=done_criteria,
)


def base_env(scenarios: str, seed=None, headless=True):
    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=[scenarios],
        agent_interfaces={"Agent": agent_interface},
        headless=headless,
        seed=seed,
    )
    return env


def base_env_v2(scenarios: str, seed=None, headless=True):
    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=[scenarios],
        agent_interfaces={"Agent": agent_interface_v2},
        headless=headless,
        seed=seed,
    )
    return env


def base_env_v3(scenarios: str, seed=None, headless=True):
    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=[scenarios],
        agent_interfaces={"Agent": agent_interface_v3},
        headless=headless,
        seed=seed,
    )
    return env


def base_env_v4(scenarios: str, seed=None, headless=True):
    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=[scenarios],
        agent_interfaces={"Agent": agent_interface_v4},
        headless=headless,
        seed=seed,
    )
    return env


def wrapped_env(
    scenarios: str,
    reward_func: str,
    include_context: bool,
    interface_version: int,
    seed=None,
    headless=True,
    max_speed=15,
):
    if interface_version == 1:
        env = base_env(scenarios, seed, headless)
    elif interface_version == 2:
        env = base_env_v2(scenarios, seed, headless)
    elif interface_version == 3:
        env = base_env_v3(scenarios, seed, headless)
    elif interface_version == 4:
        env = base_env_v4(scenarios, seed, headless)
    else:
        raise Exception("Provide valid interface version")
    env = SingleAgent(env)
    env = Api021Reversion(env)

    if include_context:
        env = ObsWrapper(env, reward_func, interface_version, max_speed)
    else:
        env = StateWrapper(env, reward_func, interface_version, max_speed)

    env = Monitor(env)
    return env


Formula_Path = "../SMARTS/smarts/env/gymnasium/wrappers/metric/formula.py"


def benchmark_env(
    scenarios: str,
    reward_func: str,
    include_context: bool,
    interface_version: int,
    seed: None,
    headless=True,
    max_speed=15,
):
    if interface_version == 1:
        env = base_env(scenarios, seed, headless)
    elif interface_version == 2:
        env = base_env_v2(scenarios, seed, headless)
    elif interface_version == 3:
        env = base_env_v3(scenarios, seed, headless)
    elif interface_version == 4:
        env = base_env_v4(scenarios, seed, headless)
    else:
        raise Exception("Provide valid interface version")
    env = Metrics(env, formula_path=Formula_Path)
    env = SingleAgent(env)
    env = Api021Reversion(env)
    if include_context:
        env = ObsWrapper(env, reward_func, interface_version, max_speed)
    else:
        env = StateWrapper(env, reward_func, interface_version, max_speed)

    env = Monitor(env)
    return env
