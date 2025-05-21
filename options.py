import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Data
    parser.add_argument('--problem', default='petra', help="The problem to solve, 'hcvrp', or 'petra' ")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--veh_num', type=int, default=3, help="The number of the problem vehicles")
    parser.add_argument('--batch_size', type=int, default=256, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=256*100, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--obj', default='min-sum', help=['min-max', 'min-sum'])

    # Model
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of heads in multi-Attention')

    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")

    # Training
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=0.995, help='Learning rate decay per epoch')
    parser.add_argument('--lr_warmup_epochs', type=int, default=5,
                        help='Number of epochs to warmup the learning rate, default 5')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=123456, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=3.0,
                        help='Maximum L2 norm for gradient clipping, default 3.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default='rollout',
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem.')

    # Misc
    parser.add_argument('--log_step', type=int, default=10, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--no_run_name_wrapper', action='store_true',
                        help='Do not add timestamp wrapper to run name')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_false', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    # Petra
    parser.add_argument('--use_data_adapter', action='store_true', help='Enable PETRA')
    parser.add_argument('--date_start', type=str, default='2025-02-18', help='Start date for the problem')
    parser.add_argument('--max_time', type=float, default=480,
                        help='Maximum allowed time for the problem')
    parser.add_argument('--cost_per_km', type=float, default=0.57 + 0.812,  # fuel + LSVA
                        help='Cost per kilometer')
    parser.add_argument('--cost_per_min', type=float, default=0.545 + 0.667,  # driver + overhead
                        help='Cost per minute')
    parser.add_argument('--petra_reward', type=str, default='critical_time_cost',
                        help='Reward type for the problem, "consumption_reward" or "critical_time_cost"')
    parser.add_argument('--consumption_reward', type=float, default=0.2, help='Cost vs reward ratio')
    # critical time cost: https://www.desmos.com/calculator/wt8y7q0cxr
    parser.add_argument('--critical_time_cost_alpha', type=float, default=1500,
                        help='Abs weight for the critical time cost')
    parser.add_argument('--critical_time_cost_beta', type=float, default=0.8,
                        help='Exp weight for the critical time cost ')
    parser.add_argument('--fulfilment', type=str, default='node_demand',
                        help='Fulfilment type for the problem, "node_demand" or "vehicle_capacity"')
    parser.add_argument('--replenishment_time_per_unit', type=float, default=0.001,
                        help='Replenishment time per unit for the problem')
    parser.add_argument('--replenishment_delay', type=float, default=10,
                        help='Replenishment delay for the problem')
    parser.add_argument('--reward_per_unit_fuel', type=float, default=0.01,
                        help='Reward per unit of fuel for the problem')
    parser.add_argument('--multi_trip', action='store_true',
                        help='Enable multi-trip for the problem')
    
    # Additional Petra parameters
    parser.add_argument('--demand_max', type=float, default=400000, 
                        help='Maximum demand value')
    parser.add_argument('--coordinates_noise_std', type=float, default=0.1,
                        help='Standard deviation of noise added to coordinates')
    parser.add_argument('--mean_speed', type=float, default=1.0,
                        help='Mean speed for travel time calculations')
    parser.add_argument('--min_critical_time', type=int, default=1,
                        help='Minimum time until critical time')
    parser.add_argument('--max_critical_time', type=int, default=40,
                        help='Maximum time until critical time')
    parser.add_argument('--demand_mu', type=float, default=50000,
                        help='Mean of demand distribution')
    parser.add_argument('--demand_sigma', type=float, default=1.0,
                        help='Standard deviation of demand distribution')
    parser.add_argument('--ttr_mu', type=float, default=5,
                        help='Mean of time-to-refill distribution')
    parser.add_argument('--ttr_sigma', type=float, default=1,
                        help='Standard deviation of time-to-refill distribution')
    parser.add_argument('--km_mu', type=float, default=90,
                        help='Mean of distance distribution')
    parser.add_argument('--km_sigma', type=float, default=0.1,
                        help='Standard deviation of distance distribution')
    parser.add_argument('--min_mu', type=float, default=90,
                        help='Mean of time distribution in minutes')
    parser.add_argument('--min_sigma', type=float, default=0.05,
                        help='Standard deviation of time distribution')
    parser.add_argument('--depot_idx', type=int, default=0,
                        help='Index of the depot node')
    
    # Wandb integration
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging (default: False)')
    parser.add_argument('--wandb_project', type=str, default='2d-petra', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity name')

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if not opts.no_run_name_wrapper:
        opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_v{}_{}".format(opts.problem, opts.veh_num, opts.graph_size),
        opts.run_name
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    assert 0 <= opts.consumption_reward <= 1, "Cost vs reward ratio must be between 0 and 1!"
    assert opts.fulfilment in ['node_demand', 'vehicle_capacity'], \
        "Fulfilment type must be either 'node_demand' or 'vehicle_capacity'!"
    assert not opts.use_data_adapter, "Data adapter is not supported for PETRA"
    return opts