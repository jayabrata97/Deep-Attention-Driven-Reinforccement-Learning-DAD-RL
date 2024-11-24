import os
import torch
import random

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import SubprocVecEnv

from feature_extractor import (
    CombinedExtractor_PPO_v2,
    CombinedExtractor_v3,
    StateExtractor_v3,
)

from env import wrapped_env
from benchmark_eval import benchmark_logger

import argparse
import yaml
import datetime
import json


# @profile
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("key", type=str, help="key for this training instance")
    args = parser.parse_args()
    key = args.key

    if not os.path.exists("./find_checkpoints"):
        os.makedirs("./find_checkpoints")

    if not os.path.isfile(f"./find_checkpoints/{key}.json"):
        with open(f"./find_checkpoints/{key}.json", "x") as f:
            data = {}
            data["checkpoint"] = None
            data["log_dir"] = None
            data["epoch"] = 0
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    else:
        with open(f"./find_checkpoints/{key}.json", "r") as f:
            data = json.load(f)
        with open(f"{data['log_dir']}/settings.yaml", "r") as f:
            cfg = yaml.safe_load(f)

    start_point = data["epoch"] + 1
    checkpoint_path = data["checkpoint"]
    log_dir = data["log_dir"]

    # PARAMETERS
    Model_name = cfg["model"]["name"]
    reward_func = cfg["model"]["reward_func"]
    include_context = cfg["model"]["include_context"]

    max_speed = cfg["action_adapter"]["max_speed"]

    scenes = cfg["env"]["scenario"]
    N_envs = cfg["env"]["num_envs"]

    modify_waypoints = False  # Only using for double merge scenario

    if "PPO" in Model_name:
        Batch_size = cfg["PPO"]["batch_size"]
        n_steps = cfg["PPO"]["n_steps"] // N_envs
        lr = cfg["PPO"]["learning_rate"]
        ent_coef = cfg["PPO"]["ent_coef"]
        target_kl = cfg["PPO"]["target_kl"]
        model_seed = cfg["PPO"]["seed"]
    elif "SAC" in Model_name:
        Batch_Size = cfg["SAC"]["batch_size"]
        LR = cfg["SAC"]["lr"]
        Learning_Starts = cfg["SAC"]["learning_starts"]
        Ent_Coef = cfg["SAC"]["ent_coef"]
        Buffer_Size = cfg["SAC"]["buffer_size"]
        model_seed = cfg["SAC"]["seed"]
        ent_scheduler = cfg["SAC"]["ent_scheduler"]

    TIMESTEPS = cfg["train"]["timesteps"]  # timesteps in one epoch
    Epochs = cfg["train"]["num_epochs"]
    DEVICE = torch.device(cfg["train"]["device"])
    TOTAL_TIMESTEPS = TIMESTEPS * Epochs
    SEED = cfg["train"]["seed"]

    eps = cfg["eval"]["episodes"]
    N_eval_envs = cfg["eval"]["num_envs"]
    Eval_seed = cfg["eval"]["num_envs"]

    model_list = ["SAC", "Attention_SAC", "PPO", "Attention_PPO"]
    if Model_name not in model_list:
        raise Exception(f"Specify a valid model name. \nValid models: {model_list}")
    else:
        pass

    include_pe = False

    interface_version = 4
    if "Attention_PPO" in Model_name:
        Attention = True
        print("\n--- Training Attention PPO Model---\n")
    elif "PPO" in Model_name:
        Attention = False
        print("\n--- Training PPO Model ---\n")
    elif "Attention_SAC" in Model_name:
        Attention = True
        print("\n--- Training Attention SAC Model ---\n")
    elif "SAC" in Model_name:
        Attention = False
        print("\n--- Training SAC Model ---\n")

    print("\n Interface version: ", interface_version)

    scenarios = {
        1: "../SMARTS/scenarios/sumo/intersections/1_to_1lane_left_turn_c_agents_1",
        2: "../SMARTS/scenarios/sumo/intersections/1_to_2lane_left_turn_t_agents_1",
        3: "../SMARTS/scenarios/sumo/SRT/left_turn",
        4: "../SMARTS/scenarios/sumo/intersections/1_to_2lane_left_turn_c_agents_1",
        5: "../SMARTS/scenarios/sumo/merge/3lane_agents_1",
        6: "../SMARTS/scenarios/sumo/straight/3lane_overtake_agents_1",
        7: "../SMARTS/scenarios/sumo/SRT/roundabout",
        8: "../SMARTS/scenarios/sumo/SRT/roundabout_medium",
        9: "../SMARTS/scenarios/sumo/SRT/roundabout_easy",
        10: "../SMARTS/scenarios/sumo/SRT/cross",
    }
    now = datetime.datetime.now()
    date = now.strftime("%d") + "-" + now.strftime("%b") + "-" + now.strftime("%y")
    time = now.strftime("%H") + "-" + now.strftime("%M") + "-" + now.strftime("%S")

    scene_name = "-".join(str(x) for x in scenes)

    if log_dir is None:
        now = datetime.datetime.now()
        date = now.strftime("%d") + "-" + now.strftime("%b") + "-" + now.strftime("%y")
        time = now.strftime("%H") + "-" + now.strftime("%M") + "-" + now.strftime("%S")
        scene_name = "-".join(str(x) for x in scenes)

        log_dir = f"logs/scenario-{scene_name}/{'state_n_context' if include_context else 'state'}/{reward_func}/{Model_name}_seed-{SEED}/{date}/{time}"

        data["log_dir"] = log_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            print(
                "\n The specified log directory already exists, this might lead to overwriting the outputs."
            )

        if SEED == "default":
            train_seeds = [42] * N_envs
        elif SEED == "mix":
            train_seeds = [random.randint(1, 100) for i in range(N_envs)]
        elif isinstance(SEED, int):
            train_seeds = [SEED] * N_envs
        else:
            raise Exception("Specify seed as 'default', 'mix', or an int")

        print(f"\n\nseeds {train_seeds} and type {type(train_seeds[0])}\n\n")

        if type(scenes) is list:
            training_scenarios = []
            for s in scenes:
                training_scenarios.append(scenarios[s])
        elif type(scenes) is int:
            training_scenarios = [scenarios[scenes]]
        else:
            raise Exception("Scenario argument must either be a list or an integer")

        cfg["training_seeds"] = train_seeds
        cfg["training_scenarios"] = training_scenarios
        with open(f"{log_dir}/settings.yaml", "w+") as f:
            yaml.dump(cfg, f)
        with open(f"./find_checkpoints/{key}.json", "w") as f:
            json.dump(data, f)
    else:
        train_seeds = cfg["training_seeds"]
        training_scenarios = cfg["training_scenarios"]

    del data
    del cfg

    print(f"\n Training Scenarios:\n {training_scenarios} \n")
    N_scenarios = len(training_scenarios)
    if N_envs % N_scenarios == 0:
        multiplier = N_envs // N_scenarios
        training_scenarios_scaled = training_scenarios * multiplier
    else:
        raise Exception(
            "Number of envs must be a mutiple of number of training scenarios"
        )

    def make_env(scenario, seed):
        env = wrapped_env(
            scenario,
            reward_func,
            include_context,
            interface_version,
            seed,
            True,
            max_speed=max_speed,
        )
        if scenario == scenarios[10]:
            modify_waypoints == True
        else:
            modify_waypoints == False
        env.waypoints_modifier(modify_waypoints)
        return env

    vec_env = SubprocVecEnv(
        [
            lambda: make_env(x, y)
            for (x, y) in zip(training_scenarios_scaled, train_seeds)
        ],
        start_method="forkserver",
    )
    assert vec_env.num_envs == N_envs

    if checkpoint_path is None:
        ### MODEL
        if include_context:
            if Attention:
                policy_kwargs = dict(
                    features_extractor_class=CombinedExtractor_v3,
                    features_extractor_kwargs=dict(pos_enc=include_pe),
                )
                print("\nCombinedExtractor_v3(Attention) is being used.\n")
            else:
                policy_kwargs = dict(features_extractor_class=CombinedExtractor_PPO_v2)
                print("\nCombinedExtractor_PPO_v2 is being used.\n")
        else:
            if Attention:
                policy_kwargs = dict(
                    features_extractor_class=StateExtractor_v3,
                    features_extractor_kwargs=dict(pos_enc=include_pe),
                )
                print("\nStateExtractor_v3 is being used.\n")
            else:
                policy_kwargs = None
                print("\nDefault feature extractor is being used.\n")

        if "PPO" in Model_name:
            model = PPO(
                "MultiInputPolicy",
                vec_env,
                batch_size=Batch_size,
                n_steps=n_steps,
                learning_rate=lr,
                ent_coef=ent_coef,
                target_kl=target_kl,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=f"{log_dir}/tensorboard",
                device=DEVICE,
                seed=model_seed,
            )
        elif "SAC" in Model_name:
            model = SAC(
                "MultiInputPolicy",
                vec_env,
                learning_rate=LR,
                buffer_size=Buffer_Size,
                learning_starts=Learning_Starts,
                batch_size=Batch_Size,
                ent_coef=Ent_Coef,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=f"{log_dir}/tensorboard",
                device=DEVICE,
                seed=model_seed,
            )
    else:
        if "PPO" in Model_name:
            model = PPO.load(checkpoint_path, vec_env, device=DEVICE)
        elif "SAC" in Model_name:
            model = SAC.load(checkpoint_path, vec_env, device=DEVICE)

        model.tensorboard_log = f"{log_dir}/tensorboard"

    print("\nModel Policy: ", model.policy)

    # CALLBACKS
    checkpoint_callback = CheckpointCallback(
        save_freq=TIMESTEPS // N_envs,
        save_path=f"{log_dir}/checkpoints",
        name_prefix=Model_name,
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=2,
    )

    class BenchmarkCallback(BaseCallback):
        """
        A custom callback that derives from ``BaseCallback``.

        :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """

        def __init__(self, verbose: int = 0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            if self.num_timesteps > 0 and self.num_timesteps % TIMESTEPS == 0:
                i = self.num_timesteps * Epochs // TOTAL_TIMESTEPS
                with open(f"./find_checkpoints/{key}.json", "r") as f:
                    data = json.load(f)
                model_path = f"{log_dir}/checkpoints/{Model_name}_{i*TIMESTEPS}_steps"

                benchmark_logger(
                    log_dir,
                    scenarios,
                    scenes,
                    model_path,
                    Eval_seed,
                    N_eval_envs,
                    reward_func,
                    include_context,
                    interface_version,
                    eps,
                    i,
                    TIMESTEPS,
                    max_speed=max_speed,
                    modify_waypoints=modify_waypoints,
                )

                data["checkpoint"] = model_path
                with open(f"./find_checkpoints/{key}.json", "w") as f:
                    json.dump(data, f)
                data["epoch"] = i
                with open(f"./find_checkpoints/{key}.json", "w") as f:
                    json.dump(data, f)
                del data

                if ent_scheduler:
                    epoch_done = self.num_timesteps // TIMESTEPS
                    decay_factor = (epoch_done + 5) // 5
                    model.ent_coef = Ent_Coef / decay_factor

            else:
                pass
            return True

    benchmark_callback = BenchmarkCallback(verbose=1)

    TRAINING_TIMESTEPS = (Epochs - start_point + 1) * TIMESTEPS
    model.learn(
        total_timesteps=TRAINING_TIMESTEPS,
        reset_num_timesteps=False,
        callback=CallbackList([checkpoint_callback, benchmark_callback]),
        progress_bar=True,
        tb_log_name="train",
    )

    vec_env.close()
    model.env.close()
    del model


if __name__ == "__main__":
    main()
