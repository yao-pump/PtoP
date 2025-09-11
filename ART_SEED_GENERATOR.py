

import random
from utility import calculate_population_distance, min_population_distance, max_population_distance

def sample_position_info(carla_map):
    NPC_TYPES = ["pedestrian"]
    v_num = random.choice([20])

    spawn_points = carla_map.get_spawn_points()
    while True:
        ego_idx = random.randint(0, len(spawn_points) - 1)
        ego_spawning = spawn_points[ego_idx]
        nearby = [sp for i, sp in enumerate(spawn_points) if i != ego_idx]
        if len(nearby) >= v_num:
            break
    selected_spawns = random.sample(nearby, v_num)
    surrounding_info = [{"transform": sp, "type": random.choice(NPC_TYPES)} for sp in selected_spawns]
    return {
        "vehicle_num": v_num,
        "ego_transform": ego_spawning,
        "surrounding_info": surrounding_info
    }

def _sanitize_position_info(pi: dict) -> dict:
    # 兼容旧字段：surrounding_transforms -> surrounding_info
    if 'surrounding_transforms' in pi and 'surrounding_info' not in pi:
        pi['surrounding_info'] = [{'transform': t, 'type': 'car'} for t in pi['surrounding_transforms']]
        pi.pop('surrounding_transforms', None)
    if 'surrounding_info' in pi:
        pi['vehicle_num'] = len(pi['surrounding_info'])
    return pi


class seed_generator:
    def __init__(self, carla_map, candidate_size):
        self.carla_map = carla_map
        self.candidate_seed_set = []
        self.executed_seed_set = []
        self.candidate_size = candidate_size



    def sample_seed(self):
        for _ in range(self.candidate_size):
            pos_info = sample_position_info(self.carla_map)
            ind = {"position_info": _sanitize_position_info(pos_info)}
            self.candidate_seed_set.append(ind)
        if len(self.executed_seed_set) == 0:
            return random.choice(self.candidate_seed_set)
        else:
            min_dist_array = []
            for seed in self.candidate_seed_set:
                min_dist_array.append(min_population_distance(seed, self.executed_seed_set))
            max_index = min_dist_array.index(max(min_dist_array))
            return self.candidate_seed_set[max_index]

    def sample_seed_random(self):
        pos_info = sample_position_info(self.carla_map)
        return {"position_info": _sanitize_position_info(pos_info)}

