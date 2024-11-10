import os
import random
from pathlib import Path
import numpy as np

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Scenario,
    Traffic,
    Flow,
    Route,
    RandomRoute,
    TrafficActor,
    SocialAgentActor,
    Distribution,
    LaneChangingModel,
    JunctionModel,
    Mission,
    EndlessMission,
    TrapEntryTactic
)

scenario = os.path.dirname(os.path.realpath(__file__))

start_routes = ["south-SN", "west-WE", "north-NS"]
end_routes = ["east-WE", "south-NS", "west-EW", "north-SN"]

# Traffic Flows
traffic = {}
for seed in np.random.choice(1000, 15, replace=False):
    actors = {}

    for i in range(4):
        car = TrafficActor(
            name = f'car_type_{i+1}',    
            speed=Distribution(mean=np.random.uniform(0.6, 1.0), sigma=0.1),
            min_gap=Distribution(mean=np.random.uniform(2, 4), sigma=0.1),
            imperfection=Distribution(mean=np.random.uniform(0.3, 0.7), sigma=0.1),
            lane_changing_model=LaneChangingModel(speed_gain=np.random.uniform(1.0, 2.0), impatience=np.random.uniform(0, 1.0), cooperative=np.random.uniform(0, 1.0)),
            junction_model=JunctionModel(ignore_foe_prob=np.random.uniform(0, 1.0), impatience=np.random.uniform(0, 1.0)),
        )

        actors[car] = 1/4

    flows = []

    for i, start in enumerate(start_routes):
        for end in end_routes:
            if end.split('-')[0] != start.split('-')[0]:
                flows.append(Flow(route=Route(begin=('edge-'+start, 0, "random"), end=('edge-'+end, 0, "random")), rate=50, actors=actors, repeat_route=True))

    traffic[str(seed)] = Traffic(flows=flows)

route = Route(begin=("edge-east-EW", 0, 1), end=("edge-south-NS", 0, "max"))
ego_missions = [
    Mission(
        route=route,
        entry_tactic=TrapEntryTactic(
            start_time=30
        )
    )
]
gen_scenario(scenario=Scenario(traffic=traffic, ego_missions=ego_missions), output_dir=Path(__file__).parent)

