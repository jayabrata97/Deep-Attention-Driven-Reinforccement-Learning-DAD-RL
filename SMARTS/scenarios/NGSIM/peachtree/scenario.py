from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"peach_{hd}",
        source_type="NGSIM",
        input_path=f"/home/airl-gpu3/Sumit_Jayabrata/Attention_planning/NGSIM_data/xy-trajectories/peach/trajectories-{hd}.txt",  # for example: f"./trajectories-{hd}.txt"
        speed_limit_mps=28,
        default_heading=0,
    )
    for hd in ["0400pm-0415pm", "1245pm-0100pm"]
]

gen_scenario(
    t.Scenario(traffic_histories=traffic_histories), output_dir=Path(__file__).parent
)
