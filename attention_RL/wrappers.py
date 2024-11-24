import gym
from gym.spaces import Box
import numpy as np
from collections import deque
import random

from data_utils import (
    get_ego_state_vector_v3,
    get_sv_state_vectors_v3,
    context_observation_adapter_v2,
)

from smarts.env.gymnasium.wrappers.metric.costs import get_dist
from smarts.core.coordinates import Point


def sparse_reward(road_map, scen, env_obs, done):

    cur_pos = Point(*env_obs["ego_vehicle_state"]["position"])
    start_pos = Point(*scen.missions["Agent"].start.position)
    end_pos = Point(*scen.missions["Agent"].goal.position)
    dist_tot = get_dist(road_map=road_map, point_a=start_pos, point_b=end_pos)
    dist_left = get_dist(road_map=road_map, point_a=cur_pos, point_b=end_pos)
    d2d = abs(dist_left / dist_tot)

    R_goal = 1 if env_obs["events"]["reached_goal"] else 0
    R_crash = -1 if env_obs["events"]["collisions"] else 0
    R_off_road = -1 if env_obs["events"]["off_road"] else 0

    speed = env_obs["ego_vehicle_state"]["speed"]

    if done:
        R_progress = 1 - d2d  # range 0-1
    else:
        R_progress = 0

    reward = (
        R_crash * (1 + 0.1 * speed)
        + R_off_road * (1 + 0.1 * speed)
        + 3 * R_goal
        + 2 * R_progress
    )

    return reward


def reward_type6_v8(road_map, scen, env_obs, done, slow_counter):

    if (env_obs["ego_vehicle_state"]["lane_id"] == "off_lane") or (
        env_obs["events"]["off_road"] == True
    ):
        speed_limit = 0
    else:
        speed_limit = road_map.lane_by_id(
            env_obs["ego_vehicle_state"]["lane_id"]
        ).speed_limit

    cur_pos = Point(*env_obs["ego_vehicle_state"]["position"])
    start_pos = Point(*scen.missions["Agent"].start.position)
    end_pos = Point(*scen.missions["Agent"].goal.position)
    dist_tot = get_dist(road_map=road_map, point_a=start_pos, point_b=end_pos)
    dist_left = get_dist(road_map=road_map, point_a=cur_pos, point_b=end_pos)
    d2d = abs(dist_left / dist_tot)

    speed = env_obs["ego_vehicle_state"]["speed"]

    R_goal = 1 if env_obs["events"]["reached_goal"] else 0
    R_crash = -1 if env_obs["events"]["collisions"] else 0
    R_off_road = -1 if env_obs["events"]["off_road"] else 0
    R_off_route = -1 if env_obs["events"]["off_route"] else 0
    R_wrong_way = -1 if env_obs["events"]["wrong_way"] else 0

    if speed < 0.1 * speed_limit:
        slow_counter += 1
    else:
        slow_counter = 0
    if slow_counter % 100 == 99:  # if vehicle is slow for continious 10s
        R_slow = -1
    else:
        R_slow = 0

    if speed_limit == 0:  # to avoid dividing by zero
        R_speed = 0
    elif speed <= speed_limit:
        R_speed = (
            (speed / speed_limit) if (R_off_route == 0 and R_wrong_way == 0) else 0
        )
    else:
        R_speed = -abs(speed - speed_limit) / speed_limit

    if done:
        R_progress = 1 - d2d  # range 0-1
        slow_counter = 0
    else:
        R_progress = 0

    reward = (
        R_crash
        + R_off_road
        + 3 * R_goal
        + 2 * R_progress
        + 0.01 * R_speed
        + 0.01 * R_off_route
        + 0.01 * R_wrong_way
        + 0.2 * R_slow
    )

    return reward, slow_counter


def action_adapter(model_action, max_speed=15):
    speed = model_action[0]  # output (0, 1)
    speed = speed * max_speed  # scale to (0, max_speed)

    speed = np.clip(speed, 0, max_speed)
    model_action[1] = np.clip(model_action[1], -1, 1)

    # discretization
    if model_action[1] < -1 / 3:
        lane = -1
    elif model_action[1] > 1 / 3:
        lane = 1
    else:
        lane = 0

    return (speed, lane)


class StateWrapper(gym.Wrapper):
    def __init__(self, env, reward_func, interface_version, max_speed):
        super().__init__(env)

        self.interface_version = interface_version
        if interface_version == 1 or interface_version == 2:
            self.action_space = gym.spaces.Box(
                low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                shape=(3,),
            )
        elif interface_version == 3:
            self.action_space = gym.spaces.Discrete(4)
        elif interface_version == 4:
            self.action_space = Box(low=np.array([0, -1]), high=np.array([1, 1]))
            self.max_speed = max_speed

        self.ego_shape = (5, 20)
        self.sv_shape = (6, 5, 6)

        self.observation_space = gym.spaces.Dict(
            {
                "sv_states": gym.spaces.Box(
                    shape=self.sv_shape, low=-np.inf, high=np.inf
                ),
                "ev_states": gym.spaces.Box(
                    shape=self.ego_shape, low=-np.inf, high=np.inf
                ),
            }
        )

        self.sv_history_queue = deque([{}] * 21, maxlen=21)
        self.reward_func = reward_func

    def step(self, action):
        if self.interface_version == 4:
            action = action_adapter(action, self.max_speed)
        env_obs, _, dones, info = self.env.step(action)

        ######################### Reward Components ##########################
        speed = env_obs["ego_vehicle_state"]["speed"]
        progress = speed * 0.1
        goal = 1 if env_obs["events"]["reached_goal"] else 0
        crash = -1 if env_obs["events"]["collisions"] else 0
        off_road = -1 if env_obs["events"]["off_road"] else 0
        off_route = -1 if env_obs["events"]["off_route"] else 0
        wrong_way = -1 if env_obs["events"]["wrong_way"] else 0
        not_moving = -1 if env_obs["events"]["not_moving"] else 0

        if self.reward_func == "dense":
            reward, self.checkpoint = reward_type6_v8(
                self.road_map, self.scen, env_obs, dones, self.slow_counter
            )
        elif self.reward_func == "sparse":
            reward = sparse_reward(self.road_map, self.scen, env_obs, dones)
        else:
            raise Exception("Must Provide a valid reward type!")

        ################################### Ego vehicle observation ##########################################
        new_ego = env_obs["ego_vehicle_state"]
        ego_state_vector, self.ev_history_queue = get_ego_state_vector_v3(
            ev_history_queue=self.ev_history_queue,
            env_obs=env_obs,
            road_map=self.road_map,
        )

        ################################# Surrounding vehicle observation ##################################
        new_sv = env_obs["neighborhood_vehicle_states"]
        self.sv_history_queue.append(new_sv)
        sv_state_vectors = get_sv_state_vectors_v3(
            self.sv_history_queue, env_obs=env_obs
        )

        obs = {"sv_states": sv_state_vectors, "ev_states": ego_state_vector}

        info["RewardComponents"] = {
            "speed": speed,
            "progress": progress,
            "goal": abs(goal),
            "crash": abs(crash),
            "off_road": abs(off_road),
            "off_route": abs(off_route),
            "wrong_way": abs(wrong_way),
            "not_moving": abs(not_moving),
        }

        info["is_success"] = env_obs["events"]["reached_goal"]
        info["is_stagnation"] = env_obs["events"]["reached_max_episode_steps"]

        return obs, reward, dones, info

    def reset(self):
        env_obs = self.env.reset()

        init_ego_vector = np.zeros(shape=(20))
        init_ego_vector[0] = 1  # padding
        self.ev_history_queue = deque([init_ego_vector] * 21, maxlen=21)
        self.sv_history_queue = deque([{}] * 21, maxlen=21)

        self.scen = self.env.unwrapped.smarts.scenario
        self.road_map = self.env.unwrapped.smarts.scenario.road_map
        self.slow_counter = 0

        ################################### Ego vehicle observation ##########################################
        new_ego = env_obs["ego_vehicle_state"]
        ego_state_vector, self.ev_history_queue = get_ego_state_vector_v3(
            ev_history_queue=self.ev_history_queue,
            env_obs=env_obs,
            road_map=self.road_map,
        )

        ################################# Surrounding vehicle observation ##################################
        new_sv = env_obs["neighborhood_vehicle_states"]
        self.sv_history_queue.append(new_sv)
        sv_state_vectors = get_sv_state_vectors_v3(
            self.sv_history_queue, env_obs=env_obs
        )

        obs = {"sv_states": sv_state_vectors, "ev_states": ego_state_vector}
        return obs

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        pass

    def waypoints_modifier(self, modify: bool):
        pass  # no need since there is context input


class ObsWrapper(gym.Wrapper):
    def __init__(self, env, reward_func, interface_version, max_speed):
        super().__init__(env)

        self.interface_version = interface_version
        if interface_version == 1 or interface_version == 2:
            self.action_space = gym.spaces.Box(
                low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                shape=(3,),
            )
        elif interface_version == 3:
            self.action_space = gym.spaces.Discrete(4)
        elif interface_version == 4:
            self.action_space = Box(low=np.array([0, -1]), high=np.array([1, 1]))
            self.max_speed = max_speed

        self.ego_shape = (5, 20)
        self.sv_shape = (6, 5, 6)
        self.context_shape = (2, 128, 128)

        self.observation_space = gym.spaces.Dict(
            {
                "sv_states": gym.spaces.Box(
                    shape=self.sv_shape,
                    low=np.float32(-np.inf),
                    high=np.float32(np.inf),
                ),
                "ev_states": gym.spaces.Box(
                    shape=self.ego_shape,
                    low=np.float32(-np.inf),
                    high=np.float32(np.inf),
                ),
                "context": gym.spaces.Box(
                    shape=self.context_shape, low=0, high=255, dtype=np.uint8
                ),
            }
        )

        self.reward_func = reward_func
        self.modify_waypoints = False

    def step(self, action):
        if self.interface_version == 4:
            action = action_adapter(action, self.max_speed)
        env_obs, _, dones, info = self.env.step(action)

        ######################### Reward Components ##########################
        speed = env_obs["ego_vehicle_state"]["speed"]
        progress = speed * 0.1
        goal = 1 if env_obs["events"]["reached_goal"] else 0
        crash = -1 if env_obs["events"]["collisions"] else 0
        off_road = -1 if env_obs["events"]["off_road"] else 0
        off_route = -1 if env_obs["events"]["off_route"] else 0
        wrong_way = -1 if env_obs["events"]["wrong_way"] else 0

        not_moving = -1 if env_obs["events"]["not_moving"] else 0

        if self.reward_func == "dense":
            ######################### Reward Adapter 6 ###########################
            reward, self.checkpoint = reward_type6_v8(
                self.road_map, self.scen, env_obs, dones, self.slow_counter
            )
        elif self.reward_func == "sparse":
            reward = sparse_reward(self.road_map, self.scen, env_obs, dones)
        else:
            raise Exception("Must Provide a valid reward type!")

        ################################### Ego vehicle observation ##########################################
        new_ego = env_obs["ego_vehicle_state"]
        ego_state_vector, self.ev_history_queue = get_ego_state_vector_v3(
            ev_history_queue=self.ev_history_queue,
            env_obs=env_obs,
            road_map=self.road_map,
        )

        ################################# Surrounding vehicle observation ##################################
        new_sv = env_obs["neighborhood_vehicle_states"]
        self.sv_history_queue.append(new_sv)
        sv_state_vectors = get_sv_state_vectors_v3(
            self.sv_history_queue, env_obs=env_obs
        )

        ################################# Context Observation ###############################
        context_states, self.goal_lane_index = context_observation_adapter_v2(
            env_obs, self.goal_lane_index, self.modify_waypoints
        )

        obs = {
            "sv_states": sv_state_vectors,
            "ev_states": ego_state_vector,
            "context": context_states,
        }

        info["RewardComponents"] = {
            "speed": speed,
            "progress": progress,
            "goal": abs(goal),
            "crash": abs(crash),
            "off_road": abs(off_road),
            "off_route": abs(off_route),
            "wrong_way": abs(wrong_way),
            "not_moving": abs(not_moving),
        }

        info["is_success"] = env_obs["events"]["reached_goal"]
        info["is_stagnation"] = env_obs["events"]["reached_max_episode_steps"]

        return obs, reward, dones, info

    def reset(self):
        env_obs = self.env.reset()
        init_ego_vector = np.zeros(shape=(20))
        init_ego_vector[0] = 1  # padding
        self.ev_history_queue = deque([init_ego_vector] * 21, maxlen=21)
        self.sv_history_queue = deque(
            [{}] * 21, maxlen=21
        )  # current timstep + last 2s of info
        self.goal_lane_index = -1

        self.scen = self.env.unwrapped.smarts.scenario
        self.road_map = self.env.unwrapped.smarts.scenario.road_map
        self.slow_counter = 0

        ################################### Ego vehicle observation ##########################################
        new_ego = env_obs["ego_vehicle_state"]
        ego_state_vector, self.ev_history_queue = get_ego_state_vector_v3(
            ev_history_queue=self.ev_history_queue,
            env_obs=env_obs,
            road_map=self.road_map,
        )

        ################################# Surrounding vehicle observation ##################################
        new_sv = env_obs["neighborhood_vehicle_states"]
        self.sv_history_queue.append(new_sv)
        sv_state_vectors = get_sv_state_vectors_v3(
            self.sv_history_queue, env_obs=env_obs
        )

        ################################# Context Observation ###############################
        context_observation_adapter_v2.counter = 0
        context_observation_adapter_v2.cons_wp = []
        context_states, self.goal_lane_index = context_observation_adapter_v2(
            env_obs, self.goal_lane_index, self.modify_waypoints
        )

        obs = {
            "sv_states": sv_state_vectors,
            "ev_states": ego_state_vector,
            "context": context_states,
        }
        return obs

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        pass

    def waypoints_modifier(self, modify: bool):
        self.modify_waypoints = modify
