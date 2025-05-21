from torch.utils.data import Dataset
import torch
import os
import pickle
from tensordict import TensorDict

class Petra:
    NAME = 'petra'
    generator = None

    def __init__(self, generator=None):
        self.generator = generator

    # @staticmethod
    def make_dataset(self, *args,**kwargs):
        # check if seed is in kwargs and generator has set_seed method
        if 'seed' in kwargs and hasattr(self.generator, 'set_seed'):
            self.generator.set_seed(kwargs['seed'])
            del kwargs['seed']
        return PetraDataset(*args,**kwargs, generator=self.generator)

def make_instance(args):
    # deprecated
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float),  # scale demand
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'capacity': torch.tensor(capacity, dtype=torch.float)
    }

class PetraDataset(Dataset):
    def __init__(self, filename=None, size=50, veh_num=3, num_samples=10000, offset=0, distribution=None, seed=None, generator=None):
        super(PetraDataset, self).__init__()

        # self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = []
            # TODO: make this load all keys
            for i in range(data['depot'].shape[0]):
                self.data.append({
                    'depot': torch.from_numpy(data['depot'][i]).float(),
                    'loc': torch.from_numpy(data['loc'][i]).float(),
                    'demand': torch.from_numpy(data['demand'][i]).float(),
                    'capacity': torch.from_numpy(data['capacity'][i]).float(),
                    'speed': torch.from_numpy(data['speed'][i]).float()
                })
        else:
            # use generator to generate data
            petra_instance = generator._generate([num_samples])
            instance = TensorDict({})
            
            # MAKE NECESSARY CHANGES
            # manually remove action_mask key
            if 'action_mask' in petra_instance:
                del petra_instance['action_mask']
            # rename some keys ["loc", "demand", "depot", "capacity", "matrix_min", "matrix_km"]
            # get depot from loc with depot_idx
            i = torch.arange(num_samples)
            instance['loc'] = torch.cat((petra_instance['coordinates'][:,:1], petra_instance['coordinates'][:,1:]), dim=1)
            instance['depot'] = petra_instance['coordinates'][i, petra_instance['depot_idx']]
            # remove depot from coordinates
            instance['demand'] = petra_instance['node_demands']
            instance['capacity'] = petra_instance['vehicle_capacity']
            instance['matrix_min'] = petra_instance['adjacency_times']
            instance['matrix_km'] = petra_instance['adjacency_km']
            instance['node_critical_time'] = petra_instance['node_critical_time']
            instance['node_consumption_rate'] = petra_instance['node_consumption_rate']
            
            self.data = PetraDataset.tdict_to_list_fast(instance)

        self.size = len(self.data)  # num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]  # index of sampled data

    @staticmethod
    def tdict_to_list_fast(tdict):
        keys = tdict.keys()
        unbound_fields = {}

        for key in keys:
            val = tdict[key]
            if isinstance(val, torch.Tensor):
                unbound_fields[key] = torch.unbind(val, dim=0)
            elif hasattr(val, 'data') and isinstance(val.data, torch.Tensor):
                unbound_fields[key] = torch.unbind(val.data, dim=0)
            else:
                # For NonTensorData or None, replicate None
                unbound_fields[key] = [None] * tdict.batch_size[0]

        batch_size = len(next(iter(unbound_fields.values())))
        return [
            {key: unbound_fields[key][i] for key in keys}
            for i in range(batch_size)
        ]