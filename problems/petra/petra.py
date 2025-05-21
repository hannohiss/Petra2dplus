from options import get_options
import torch

class PetraEnv:
    """
    Environment for the PetraRL algorithm adjusted to work with 2D-Ptr implementation.
    Args:
        generator from petra_rl that generates the problem instance.
            - 'loc': Customer coordinates [batch_size, graph_size, 2]
            - 'demand': Customer demands [batch_size, graph_size]
            - 'depot': Depot coordinates [batch_size, 2]
            - 'capacity': Vehicle capacities [batch_size, vehicle_num]
            - 'adjacency_times': Adjacency matrix for travel times [batch_size, graph_size, graph_size]
            - 'adjacency_km': Adjacency matrix for distances [batch_size, graph_size, graph_size]
    """
    def __init__(self, input, opts=None):
        assert opts is not None, "Options must be provided"
        self.vehicles = opts.vehicles
        self.demand_max = opts.demand_max
        self.vehicle_max_capacity = max([veh['load'] for veh in self.vehicles])
        self.max_critical_time = opts.max_critical_time
        self.min_critical_time = opts.min_critical_time
        self.batch_size = input["loc"].shape[0]
        self.bs_index = torch.arange(self.batch_size)
        self.step = 0
        self.max_time = opts.max_time
        self.cost_per_km = opts.cost_per_km
        self.cost_per_min = opts.cost_per_min
        self.consumption_reward = opts.consumption_reward
        self.critical_time_cost_alpha = opts.critical_time_cost_alpha
        self.critical_time_cost_beta = opts.critical_time_cost_beta
        self.petra_reward = opts.petra_reward
        self.fulfilment = opts.fulfilment
        self.replenishment_time_per_unit = opts.replenishment_time_per_unit
        self.replenishment_delay = opts.replenishment_delay
        self.reward_per_unit_fuel = opts.reward_per_unit_fuel
        self.multi_trip = opts.multi_trip
        self.initial_node_state(
            input["loc"], 
            input["demand"], 
            input["depot"], 
            input["node_critical_time"], 
            input["node_consumption_rate"]
        )
        self.initial_veh_state(input["capacity"])
        self.matrix_min = input["matrix_min"]
        self.matrix_km = input["matrix_km"]

    def initial_node_state(self, loc, demand, depot, node_critical_time, node_consumption_rate):
        """
        Initialize the node state with customer coordinates and demands.
        Args:
            loc (torch.Tensor): Customer coordinates [batch_size, graph_size, 2].
            demand (torch.Tensor): Customer demands [batch_size, graph_size].
            depot (torch.Tensor): Depot coordinates [batch_size, 2].
        """
        assert (
            loc.shape[:2] == demand.shape
        ), "The customer's loc and demand shape do not match"
        self.customer_num = loc.shape[1] - 1
        # IMPORTANT: depot is already included
        # self.coords = torch.cat([depot, loc], dim=1)
        self.N = loc.shape[1] # + 1
        self.coords = loc
        self.demand = demand
        self.demand_satisfied = torch.zeros_like(demand)
        self.node_critical_time = node_critical_time
        self.node_consumption_rate = node_consumption_rate
        self.visited = torch.zeros_like(self.demand).bool()
        self.visited[:, 0] = True

    def all_finished(self):
        return self.finished().all()

    def finished(self):
        '''
        Finishing criteria
        1. All vehicles are back at depot 
        AND
          a) have no capacity left
          OR
          b) have no time left
        '''
        # Check if all vehicles are at depot
        all_vehicles_at_depot = self.veh_cur_node == 0
        # Check if all vehicles are empty
        all_vehicles_empty = (self.veh_capacity - self.veh_used_capacity) <= 0
        # Check if all vehicles have no time left
        all_vehicles_no_time = self.veh_time >= self.max_time
        # all vehicles moved
        all_vehicles_moved = self.veh_time > 0
        # Those that have not moved are at depot
        all_nodes_visited = self.visited.all(-1).unsqueeze(-1)
        
        all_vehicles_finished = (
            (all_vehicles_at_depot & all_vehicles_empty)
            | (all_vehicles_at_depot & all_vehicles_no_time) 
            # | (all_vehicles_at_depot & all_vehicles_moved)
            | (all_vehicles_at_depot & all_nodes_visited)
        )
        return all_vehicles_finished.all(-1)
    

    def get_all_node_state(self):
        """Returns: torch.Tensor: [bs, N, 5], get node initial features."""
        return torch.cat(
            [
                self.coords,
                self.demand.unsqueeze(-1)/(self.demand_max//20),
                self.node_critical_time.unsqueeze(-1)/self.max_critical_time,
                self.node_consumption_rate.unsqueeze(-1)/((self.demand_max//20) / self.min_critical_time),
            ],
            dim=-1
        )

    def initial_veh_state(self, capacity):
        """
        Initialize the vehicle state with capacities.
        Args:
            capacity (torch.Tensor): Vehicle capacities [batch_size, vehicle_num].
        """
        self.veh_capacity = capacity
        self.veh_num = capacity.shape[1]
        self.veh_time = torch.zeros_like(capacity)
        self.veh_cur_node = torch.zeros_like(capacity).long()
        self.veh_used_capacity = torch.zeros_like(capacity)
        self.veh_cost = torch.zeros_like(capacity)
        self.veh_index = torch.arange(self.veh_num)

    def get_all_veh_state(self):
        """Returns: torch.Tensor: [bs, veh_num, 5]."""
        veh_cur_coords = self.coords[self.bs_index.unsqueeze(-1),self.veh_cur_node]
        # TODO: check if unsqueeze is needed
        return torch.cat([
            veh_cur_coords,
            self.veh_time.unsqueeze(-1)/self.max_time,
            self.veh_capacity.unsqueeze(-1)/self.vehicle_max_capacity,
            self.veh_used_capacity.unsqueeze(-1)/self.vehicle_max_capacity,
        ],dim=-1)

    def update(self, veh, next_node, fulfilment):
        """
        Update the environment with the selected vehicle and node.
        Args:
            veh [bs] (torch.Tensor): Selected vehicle indices. 
            next_node [bs] (torch.Tensor): Selected node indices.
            fulfilment [bs] (torch.Tensor): How much demand is fulfilled.
        """
        # select node must be unvisited, except depot
        assert not self.visited[self.bs_index, next_node][
            next_node != 0
        ].any(), "Wrong solution: node has been selected !"

        # LAST LOCATION
        depot_mask = next_node == 0
        last_node = self.veh_cur_node[self.bs_index, veh]

        # FULFILMENT
        # we can only drop the demand, if the vehicle has enough capacity
        if self.fulfilment == "node_demand":
            drop = self.demand[self.bs_index, next_node] * fulfilment
        elif self.fulfilment == "vehicle_capacity":
            drop = self.veh_capacity[self.bs_index, veh] * fulfilment
        else:
            raise ValueError("Unknown fulfilment type: {}".format(self.fulfilment))    
        drop = torch.min(
            # only drop what fulfilment predicts
            drop,
            torch.min(
                # can only drop what is left
                self.veh_capacity[self.bs_index, veh] - self.veh_used_capacity[self.bs_index, veh],
                # can only drop what is demanded
                self.demand[self.bs_index, next_node] - self.demand_satisfied[self.bs_index, next_node]
            )
        ).clamp(min=0)
        self.veh_used_capacity[self.bs_index, veh] += drop
        self.demand_satisfied[self.bs_index, next_node] += drop

        # TIME
        travel_time = self.matrix_min[self.bs_index, last_node, next_node]  # [bs, veh_num, 2]
        drop_time = drop * self.replenishment_time_per_unit + self.replenishment_delay
        # self.veh_time[self.bs_index, veh] += travel_time + torch.where(depot_mask, torch.zeros_like(drop_time), drop_time)
        refill_time = torch.zeros_like(drop_time) # default

        # Calculate refill time based on how much needs to be refilled
        if self.multi_trip and depot_mask.any():
            refill_amount = self.veh_used_capacity[self.bs_index, veh][depot_mask]
            refill_time[depot_mask] = refill_amount * self.replenishment_time_per_unit + self.replenishment_delay
            self.veh_used_capacity[self.bs_index, veh][depot_mask] = 0
        # Add travel time plus either refill time (at depot) or drop time (at customer)
        self.veh_time[self.bs_index, veh] += travel_time + torch.where(depot_mask, refill_time, drop_time)

        # COST
        self.veh_cost[self.bs_index, veh] += (travel_time + torch.where(depot_mask, refill_time, drop_time)) * self.cost_per_min
        self.veh_cost[self.bs_index, veh] += self.matrix_km[self.bs_index, last_node, next_node] * self.cost_per_km

        # NEW LOCATION
        self.veh_cur_node[self.bs_index, veh] = next_node

        # VISITED
        self.visited[self.bs_index, next_node] = True
        
        self.step += 1

        return drop

    # WIP: NEED TO ADD BACK_TO_DEPOT FUNCTION WHEN TIME IS UP
    # NOT NECESSARILY IF DONE BY ACTION MASK

    def get_action_mask(self):
        """
        IMPORTANT: True means the action is not allowed.
        Returns: torch.Tensor: [bs, M, N+1], action mask. 
        """
        # 1. Visited nodes
        visited_mask = self.visited.clone()
        visited_mask[:, 0] = False  # depot is not visited
        visited_mask = (
            visited_mask.unsqueeze(1)
            .expand(self.batch_size, self.veh_num, self.N)
            .clone()
        )  # [bs, veh_num, N+1]
        # Vehicle cannot stay in place to avoid visiting the depot twice,
        visited_mask[
            self.bs_index.unsqueeze(-1),
            self.veh_index.unsqueeze(0),
            self.veh_cur_node
        ] = True

        # 2. Empty vehicles, need to go back to depot (capacity constraint)
        # Empty vehicle mask - calculate for each vehicle and each potential node
        empty_veh = (
            self.veh_capacity - self.veh_used_capacity
        ) <= 0
        empty_veh_mask = empty_veh.unsqueeze(-1).expand(
            self.batch_size, self.veh_num, self.N
        )

        # 3. Vehicles that have no time left (time constraint)
        # # Time mask - calculate for each vehicle and each potential node
        # veh_time_mask = torch.zeros_like(visited_mask, dtype=torch.bool)
        # # Get the batch indices for the current batch
        # batch_indices = self.bs_index.unsqueeze(-1).expand(self.batch_size, self.veh_num)
        # # Get the time to depot for all vehicles
        # time_to_depot = self.matrix_min[
        #     batch_indices, self.veh_cur_node, torch.zeros_like(self.veh_cur_node)
        # ]
        # Check if the time to depot is greater than the max time
        
        # 3.1 Start easy with masking only violations
        veh_time = (self.veh_time > self.max_time)
        veh_time_mask = veh_time.unsqueeze(-1)

        # SPECIAL CASE
        mask = torch.ones_like(visited_mask, dtype=torch.bool)
        # because in batch processing the finished task will have a full mask and raise an error
        mask[self.finished(),0,0] = False  # (veh 0, stays at depot)
        # allow depot for empty vehicles, only if they are not at depot
        batch_idx, row_idx = ((self.veh_cur_node != 0) & empty_veh).nonzero(as_tuple=True)
        mask[batch_idx, row_idx, 0] = False
        # if vehicle is not at depot and has no time left, it can violate the time constraint
        batch_idx, row_idx = ((self.veh_cur_node != 0) & veh_time).nonzero(as_tuple=True)
        mask[batch_idx, row_idx, 0] = False

        # 4. Combine all masks
        final_mask = mask & (visited_mask | empty_veh_mask | veh_time_mask)
        
        return final_mask

    def all_go_depot(self):
        """
        DEPRECATED
        All vehicles go back to the depot.
        """
        veh_list = torch.arange(self.veh_num)
        depot = torch.zeros_like(self.bs_index)
        for i in veh_list:
            self.update(i.expand(self.batch_size), depot)
            self.veh_time[self.bs_index, i] += self.matrix_min[
                self.bs_index, self.veh_cur_node, 0
            ]
            self.veh_cost[self.bs_index, i] += (
                self.matrix_min[self.bs_index, self.veh_cur_node, 0] * self.cost_per_min
            ) + (self.matrix_km[self.bs_index, self.veh_cur_node, 0] * self.cost_per_km)

    def get_cost(self,obj):
        """Returns: torch.Tensor: Cost of the current solution."""
        
        if self.petra_reward == "consumption_reward":
            node_cost = (self.demand_satisfied * self.node_consumption_rate)
        elif self.petra_reward == "critical_time_cost":
            new_node_critical_time = self.node_critical_time + (self.demand_satisfied / self.node_consumption_rate.clamp(min=1e-6))
            # minus to make it a reward
            node_cost = self.critical_time_cost_alpha * torch.exp(-new_node_critical_time[:,1:] * self.critical_time_cost_beta)
        else:
            node_cost = torch.zeros_like(self.demand_satisfied)

        if obj == "min-sum":
            veh_cost = self.veh_cost.sum(-1)
            node_cost = node_cost.sum(-1)
        elif obj == "min-max":
            # minimize max cost
            veh_cost = self.veh_cost.max(-1)
            # maximize min reward
            node_cost = node_cost.min(-1)
        else:
            raise ValueError("Unknown objective function: {}".format(obj))
        
        # we dont want vehicles to be non-empty at depot
        depot_cost = (self.veh_capacity - self.veh_used_capacity).sum(-1)

        # revenue due to fulfilled demand
        revenue = self.demand_satisfied.sum(-1) * self.reward_per_unit_fuel

        # cost vs reward
        cost = veh_cost + depot_cost + node_cost * self.consumption_reward - revenue

        return cost

    @classmethod
    def from_data(cls, loc, demand, depot, capacity, matrix_min, matrix_km):
        """
        Create an environment directly from data without using a generator.
        This is used during evaluation with pre-generated datasets.
        """
        # Create a dummy instance without a generator
        instance = cls.__new__(cls)

        # Set up the environment
        batch_size = 1 if len(loc.shape) == 2 else loc.shape[0]

        instance.device = loc.device
        instance.batch_size = batch_size
        instance.bs_index = torch.arange(batch_size)
        instance.step = 0

        # Set default values for parameters that would normally come from generator
        instance.max_time = 480  # Default value
        instance.cost_per_km = 1.382  # Default value (0.57 + 0.812)
        instance.cost_per_min = 1.212  # Default value (0.545 + 0.667)

        # Initialize state
        instance.initial_node_state(loc, demand, depot)
        instance.initial_veh_state(capacity)

        # Set the travel matrices
        instance.matrix_min = matrix_min
        instance.matrix_km = matrix_km

        return instance
