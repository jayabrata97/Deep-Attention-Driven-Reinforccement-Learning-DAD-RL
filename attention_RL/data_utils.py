import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im 
import sys
from smarts.core.utils.math import position_to_ego_frame
import time

def get_ego_state_vector_v3(ev_history_queue, env_obs, road_map) -> np.array:

    ego_obs = env_obs['ego_vehicle_state']

    if (ego_obs['lane_id']=="off_lane") or (env_obs['events']['off_road']==True):
        lane_speed_limit = 0
    else:
        lane_speed_limit = road_map.lane_by_id(ego_obs['lane_id']).speed_limit

    ego_padding = np.array([0])
    ego_state_vector = np.concatenate((
        ego_padding,
        ego_obs['position'][:2],
        ego_obs['heading'],
        ego_obs['speed'],
        ego_obs['lane_index'],

        lane_speed_limit,
        ego_obs['mission']['goal_position'][:2],

        ego_obs['steering'],
        ego_obs['yaw_rate'],
        ego_obs['linear_velocity'][:2],
        ego_obs['angular_velocity'][2],
        ego_obs['linear_acceleration'][:2],
        ego_obs['angular_acceleration'][2],
        ego_obs['linear_jerk'][:2],
        ego_obs['angular_jerk'][2],
        ), axis=None)

    ego_state_vector = np.array([ego_state_vector.reshape((ego_state_vector.size))]) # convert (20,) to (1,20)
    ego_state_vector = ego_state_vector.astype('float32')

    ev_history_queue.append(ego_state_vector.squeeze())

    # ego-vehicle observation
    ego_states = np.zeros((5,20))
    for i in range(5):
        ego_states[i] = ev_history_queue[5*i]

    ego_states.astype(np.float32, copy=False)

    return ego_states, ev_history_queue

def get_sv_state_vectors_v3(sv_history_queue, env_obs) -> np.array:

    #sv_history_queue length is 21 (2.1s)
    history_sub = [sv_history_queue[0], sv_history_queue[5], sv_history_queue[10], sv_history_queue[15], sv_history_queue[20]]

    sv_dict = {}
    sv_state_history_vector_dims = (5, 6)

    ### sv_obs is a dictionary of concatenatd features for all vehicles together, for example 'sv_obs.position' is an array of positions of N surrounding vehicles.
    for idx, sv_obs in enumerate(history_sub):
        if sv_obs=={}:
            continue
        for i in range(len(sv_obs['id'])):
            sv_padding = np.array([0])      #denotes presence of vehicle
            new_sv_state = np.concatenate((
                sv_padding,
                sv_obs['position'][i][:2],
                sv_obs['heading'][i],
                sv_obs['speed'][i],
                sv_obs['lane_index'][i],
            ), axis=None)
            if sv_obs['id'][i] == '': 
                break
            elif sv_obs['id'][i] not in sv_dict:
                sv_state_history_vector = np.zeros(sv_state_history_vector_dims)
                sv_state_history_vector[:, 0] = 1       #padding 1 means vehicle not present
                sv_dict[sv_obs['id'][i]] = sv_state_history_vector
            sv_dict[sv_obs['id'][i]][idx, :] = new_sv_state
            
    dist_dict = {}
    ego_pos = np.array(env_obs['ego_vehicle_state']['position'][:2])
    for id in sv_dict:  # making a dictionary with the vehicle ids and their relative distances from the ego based on current timestep
        curr_state = sv_dict[id][-1,:].squeeze()
        sv_pos = np.array(curr_state[1:3])  #first feature is padding
        if curr_state[0]==1:   # when the vehicle is not present in the current timestep
            dist = np.inf
        elif curr_state[0]==0:
            dist = np.linalg.norm(sv_pos - ego_pos)
        dist_dict[id] = dist
    
    sorted_dist_dict = dict(sorted(dist_dict.items(), key=lambda i:i[1]))   # sorting the dict based on the values
    filtered_ids = list(sorted_dist_dict.keys())[:6]  # clipping nearest 6 vehicles (if <=6 then no change)           
    filtered_sv_dict = {}
    for id in filtered_ids:
        filtered_sv_dict[id] = sv_dict[id]  # obtain the sv_states based on the clipped ids

    #stack all the vehicle data
    sv_list = [i for i in filtered_sv_dict.values()]
    temp = np.zeros((6,5,6))        #(vehicles, timesteps, features)
    temp[:,:,0] = 1    #indicates that vehicle is absent (to be learned by the model)
    if len(sv_list)==0:
        sv_state_vectors = temp.astype('float32')
    else:
        sv_stack = np.stack(sv_list, axis=0)     #shape = (N,5,6)
        s = sv_stack.shape[0]
        temp[:s, :, :] = sv_stack
        sv_state_vectors = temp.astype('float32')

    return sv_state_vectors

def context_observation_adapter_v2(env_obs, goal_lane_index, modify_waypoints:bool) -> np.array:
    """
    Context encoder for RL state encoding
    Args:
        env_obs (<class 'smarts.core.sensors.Observation'>): default environment observation
    Returns:
        np.array: Final combined state encoding for RL using Context encoding and attention mechanism between ego and surrounding vehicles
    """
    #only dgm is used for now
    new_dgm = env_obs['drivable_area_grid_map']     #128x128x1
    new_dgm = np.moveaxis(new_dgm, -1, 0)           #1x128x128
    context_observation_adapter_v2.counter += 1

    if modify_waypoints == False:
    
        ego_lane = env_obs['ego_vehicle_state']['lane_index']   # waypoints only collected for the lane that agent is following
        waypoints_global = env_obs['waypoint_paths']['position'][ego_lane, :50, :] # in global coordinates, probably 12 lanes.
        waypoints_global = np.array(waypoints_global)

        goal_pos = env_obs['ego_vehicle_state']['mission']['goal_position']
        ego_pos = env_obs['ego_vehicle_state']['position']
        dist = int(np.linalg.norm(ego_pos-goal_pos))
        if dist<=30:
            if goal_lane_index==-1:   #find goal_lane
                for lane_index in range(len(env_obs['waypoint_paths']['position'])):
                    for w in env_obs['waypoint_paths']['position'][lane_index]:
                        if np.linalg.norm(goal_pos-w) < 0.5:
                            goal_lane_index = lane_index
                            break
            else:
                waypoints_global[:50] = env_obs['waypoint_paths']['position'][goal_lane_index,:50]

    elif modify_waypoints == True:
        waypoints_global = env_obs['waypoint_paths']['position'][:1, :50, :]
        waypoints_global = np.array(waypoints_global)
        
        """ ---------------------- New Addition: Manual Waypoint Generation for Double Merge --------------------"""

        if len(context_observation_adapter_v2.cons_wp) < 2:
            context_observation_adapter_v2.cons_wp.append(waypoints_global[0, 0, :])
        else:
            context_observation_adapter_v2.cons_wp[0] = context_observation_adapter_v2.cons_wp[1]
            context_observation_adapter_v2.cons_wp[1] = waypoints_global[0, 0, :]

        num_waypoints = 25

        if len(context_observation_adapter_v2.cons_wp) == 2 and np.sum(waypoints_global[0, 1, :]) == .0:
            curr_wp = context_observation_adapter_v2.cons_wp[1]
            prev_wp = context_observation_adapter_v2.cons_wp[0]

            if (curr_wp[:2] != prev_wp[:2]).all():
                gen_waypoints_global = np.zeros((num_waypoints, 3)) 

                dist_waypoint_set = 1 # set distance between two waypoints
                slope_waypoint = (curr_wp[1] - prev_wp[1])/(curr_wp[0] - prev_wp[0]) # this is m

                # generate next waypoint
                move_x = (dist_waypoint_set/(1 + slope_waypoint**2))**(1/2)
                move_y = slope_waypoint*move_x
                mul_matrix = np.linspace(start=1, stop=25, num=25)
                gen_waypoints_global[:, 0] = mul_matrix*move_x + curr_wp[0]
                gen_waypoints_global[:, 1] = mul_matrix*move_y + curr_wp[1] 
                waypoints_global = gen_waypoints_global

        waypoints_global = waypoints_global.reshape((-1,waypoints_global.shape[-1]))


    """ ---------------------- New Addition: Manual Waypoint Generation for Double Merge --------------------"""

    waypoints_local = np.zeros(waypoints_global.shape)
    ego_global = env_obs['ego_vehicle_state']
    mtr_to_pix_scale = 128/50
    waypoints_pix_local = np.zeros(waypoints_global.shape)
    
    for i in range(waypoints_global.shape[0]):
        test_ego_points = position_to_ego_frame(ego_global['position'], ego_global['position'], ego_global['heading'])
        waypoints_local[i] = position_to_ego_frame(waypoints_global[i], ego_global['position'], ego_global['heading'])
        waypoints_pix_local[i] = waypoints_local[i]*mtr_to_pix_scale

    # should convert to ego-coordinates and convert to pixel coordinates
    route_map = np.zeros((128, 128))

    waypoints_img = np.zeros(waypoints_pix_local[:,:2].shape)
    for i in range(waypoints_pix_local.shape[0]):
        waypoints_img[i] = waypoints_pix_local[i,:2] - np.array([-64, 64]) # translation
        waypoints_img[i] = waypoints_img[i]@np.array([[0, 1], [-1, 0]]) # rotation
    
    # generate gaussian patch
    def gaussian_heatmap(center, sig=0.04):
        image_size = (128, 128)
        x_axis = np.linspace(0, image_size[0]-1, image_size[0]) - center[0]
        y_axis = np.linspace(0, image_size[1]-1, image_size[1]) - center[1]
        xx, yy = np.meshgrid(x_axis, y_axis)
        kernel = np.exp(-0.5 * (np.square(xx) +
                                np.square(yy)) / np.square(sig))
        if np.max(kernel) == 0:
            kernel = np.zeros(shape=image_size)
        else:
            kernel = kernel*255

        return kernel
    
    for i in range(waypoints_img.shape[0]):
        patch = gaussian_heatmap(np.array([waypoints_img[i][0], waypoints_img[i][1]]), sig=2)
        route_map = route_map + patch

    if route_map.max()>0:
        route_map = route_map/route_map.max()*255 #scale between 0 to 255 to match standard image space

    route_map = route_map.T # to match DGM 
    route_map = route_map.reshape((1,128,128))

    context_states = np.concatenate((new_dgm, route_map), axis=0)
    context_states.astype(np.uint8, copy=False)

    return context_states, goal_lane_index
