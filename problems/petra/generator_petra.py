import math
from typing import Dict, List

import torch
from tensordict.tensordict import TensorDict


class PetraGeneratorRandom():
    """
    Class to generate instances of the PETRA problem.

    Args:
        opts: Options containing problem parameters
        num_loc: Number of locations in the problem
        vehicles: List of vehicle capacities
    """

    def __init__(
        self,
        opts,
        num_loc: int = 28,
        vehicles: List[Dict[str, float]] = [{"load": 20000},{"load": 35000}] + [{"load": 40000}] * 5,
    ) -> None:
        
        # Node related parameters
        self.num_loc = num_loc
        if vehicles is None:
            vehicles = [
                {"load": 2},
                {"load": 3},
                {"load": 5},
                {"load": 5},
                {"load": 10},
            ]
        self.num_vehicles = len(vehicles)
        self.vehicle_capacity = torch.tensor(
            [vehicle["load"] for vehicle in vehicles], dtype=torch.float32
        )
        self.max_vehicle_capacity = self.vehicle_capacity.max()
        
        # Initialize parameters from opts
        self.cost_per_km = opts.cost_per_km
        self.cost_per_min = opts.cost_per_min
        self.replenishment_delay = opts.replenishment_delay
        self.replenishment_time_per_unit = opts.replenishment_time_per_unit
        self.reward_per_unit_fuel = opts.reward_per_unit_fuel
        self.max_time = opts.max_time
        self.demand_max = opts.demand_max
        self.coordinates_noise_std = opts.coordinates_noise_std
        self.mean_speed = opts.mean_speed
        self.min_critical_time = opts.min_critical_time
        self.max_critical_time = opts.max_critical_time
        self.depot_idx = opts.depot_idx
        self.demand_mu = opts.demand_mu
        self.demand_sigma = opts.demand_sigma
        self.ttr_mu = opts.ttr_mu
        self.ttr_sigma = opts.ttr_sigma
        self.km_mu = opts.km_mu
        self.km_sigma = opts.km_sigma
        self.min_mu = opts.min_mu
        self.min_sigma = opts.min_sigma

    def set_seed(self, seed: int) -> None:
        """
        Set the seed for random number generation.

        Args:
            seed (int): The seed value to set.
        """
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _generate(self, batch_size: torch.Size) -> TensorDict:
        batch_ids = torch.arange(*batch_size)

        # Step 1: Add Gaussian noise
        noise = torch.randn(*batch_size, self.num_loc, 2) * (self.coordinates_noise_std**0.5)
        coordinates = torch.clamp(noise + 0.5, 0.0, 1.0)  # [B, N, 2]

        # Step 2: Recompute pairwise distances
        diffs = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)  # [B, N, N, 2]
        dists = torch.norm(diffs, dim=-1)  # [B, N, N]

        # Step 3: Rescale distances to match original average scale
        adjacency_km = dists * self.km_mu
        adjacency_km = adjacency_km + (torch.randn_like(adjacency_km) * self.km_sigma).clamp(min=-self.km_sigma, max=self.km_sigma)

        # Step 4: Recompute times + add 10% noise
        adjacency_times = dists * self.min_mu
        adjacency_times = adjacency_times + (torch.randn_like(adjacency_times) * self.min_sigma).clamp(min=-self.min_sigma, max=self.min_sigma)

        # diagonal is 0
        diag_indices = torch.arange(self.num_loc)
        adjacency_times[:, diag_indices, diag_indices] = 0.0
        adjacency_km[:, diag_indices, diag_indices] = 0.0

        # Step 5: Generate node demands
        demand_dist = LogNormalDistribution(math.log(self.demand_mu), self.demand_sigma)
        node_demands = demand_dist.sample((*batch_size, self.num_loc)).clamp(max=self.demand_max)
        node_demands[batch_ids, self.depot_idx] = 0.0 # depot has no demand

        # step 6: Generate node_critical_time
        ttr_dist = LogNormalDistribution(math.log(self.ttr_mu), self.ttr_sigma)
        node_critical_time = ttr_dist.sample((*batch_size, self.num_loc)).clamp(min=self.min_critical_time, max=self.max_critical_time)
        # make sure that the node_critical_time is at least 2 days of the demand
        node_critical_time = node_critical_time.clamp(min=node_demands * 2 / self.max_vehicle_capacity)
        node_critical_time[batch_ids, self.depot_idx] = self.max_critical_time # math.inf weird bug

        # compute node_consumption_rate
        node_consumption_rate = node_demands / node_critical_time # [B, N]

        # demands should not be satisfiable with the given total vehicle capacity
        mask = (node_demands.sum(dim=-1) <= self.vehicle_capacity.sum())
        if mask.any():
            add = mask * ((self.vehicle_capacity.sum()-node_demands.sum(dim=-1).min())/self.num_vehicles+1e-3)
            # add to all nodes 
            node_demands += add.unsqueeze(-1)
        
        assert (node_demands.sum(dim=-1) > self.vehicle_capacity.sum()).all(), "Demands are satisfiable with the given vehicle capacity"

        # Put all variables together
        return TensorDict(
            {
                "adjacency_km": adjacency_km,  # [batch, nodes, nodes]
                "adjacency_times": adjacency_times,  # [batch, nodes, nodes]
                "coordinates": coordinates,  # [batch, nodes, 2]
                "node_critical_time": node_critical_time, # [batch, nodes]
                "node_demands": node_demands, # [batch, nodes] 
                "node_consumption_rate": node_consumption_rate, # [batch, nodes]
                "vehicle_capacity": self.vehicle_capacity.unsqueeze(0).expand((*batch_size, self.num_vehicles)),  # [batch, vehicles]
                "depot_idx": torch.full((*batch_size,), self.depot_idx, dtype=torch.int32),  # [batch]
            },
            batch_size=batch_size,
        )
    

class LogNormalDistribution():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self, size):
        return torch.exp(torch.randn(size) * self.sigma + self.mu)
