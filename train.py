import copy
import os
import pprint as pp
import time
from tqdm import tqdm
import torch
import math
import wandb

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import make_gif, log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts, step=0):
    # Validate
    print('Validating...')
    # multi batch
    cost, infos, actions = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    

    # calculate KPIs
    episode_lengths = [action['node'].size(0) for action in actions]
    min_episode_len = min(episode_lengths)
    max_episode_len = max(episode_lengths)
    # make all episode lengths max_episode_len by padding with zeros
    for action in actions:
        if action['node'].size(0) < max_episode_len:
            action['node'] = torch.cat([action['node'], torch.zeros(max_episode_len - action['node'].size(0))], dim=0)
            action['veh'] = torch.cat([action['veh'], torch.zeros(max_episode_len - action['veh'].size(0))], dim=0)
            action['fulfilment'] = torch.cat([action['fulfilment'], torch.zeros(max_episode_len - action['fulfilment'].size(0))], dim=0)
    # assert min_episode_len == max_episode_len, "All episode lengths must be the same, this might be due to batching (try increasing batch size)"
    node_actions = torch.stack([action['node'] for action in actions], dim=0)
    veh_actions = torch.stack([action['veh'] for action in actions], dim=0)
    fulfilment_actions = torch.stack([action['fulfilment'] for action in actions], dim=0)
    all_actions = torch.stack([node_actions, veh_actions, fulfilment_actions], dim=1)

    # KPIs: 1. Trip to drop ratio 2. Trip to drop ratio per vehicle type
    num_vehicles = len(opts.vehicles)
    trips_home = torch.where((node_actions == 0) & (veh_actions == 0), 0.0, 1.0)
    trips_home_vehicles = torch.where((node_actions == 0) & (veh_actions == 0), -1, veh_actions)
    stops_per_trip_ratio_veh = []
    for i in range(num_vehicles):
        trips_to_drop = torch.where(trips_home_vehicles == i, 1.0, 0.0)
        stops_per_trip_ratio_veh.append(trips_to_drop.sum(dim=1).mean(dim=0) - (0 if i == 0 else 1))

    vehicles = [{**veh, 'stops_per_trip': ratio, 'drop_per_stop': veh['load'] / ratio} for veh, ratio in zip(opts.vehicles, stops_per_trip_ratio_veh)]

    trips_to_drop = trips_home.sum(dim=1) - num_vehicles
    stops_per_trip_ratio = trips_to_drop.mean() / num_vehicles
    
    # average drop-off

    stats = {
        'validation/avg_cost': avg_cost.item(),
        'validation/std_cost': torch.std(cost).item() / math.sqrt(len(cost)),
        'validation/stops_per_trip_ratio': stops_per_trip_ratio.item(),
    }
    for i, veh in enumerate(vehicles):
        stats[f'validation/stops_per_trip/vehicle_{i}'] = veh['stops_per_trip'].item()
        stats[f'validation/drop_per_stop/vehicle_{i}'] = veh['drop_per_stop'].item()
        stats[f'validation/load/vehicle_{i}'] = veh['load']

    # Log validation metrics to wandb
    if hasattr(opts, 'use_wandb') and opts.use_wandb and wandb.run is not None:
        wandb.log(stats, step=step)
    else:
        pp.pprint(stats)

    # info is a list of dicts
    # each dict has keys: inner_info, cost_info
    #    inner_info is a list of dicts
    #    cost_info is a list of dicts with keys node_cost, depot_cost, veh_cost, revenue
    num_infos = len(infos)
    keys = infos[0]['cost_info'].keys()
    data = {key: [] for key in keys}
    for i in range(num_infos):
        for key in keys:
            data[key].append(infos[i]['cost_info'][key].mean().item())
    if hasattr(opts, 'use_wandb') and opts.use_wandb and wandb.run is not None:
        for key in keys:
            wandb.log({
                f'validation/{key}_mean': torch.tensor(data[key]).mean().item(),
                f'validation/{key}_std': torch.tensor(data[key]).std().item()
            }, step=step)


    # Generate GIFs and log them to wandb
    for i, bat in enumerate(DataLoader(dataset, batch_size=1)):
        with torch.no_grad():
            cost, log_likelihood, pi, veh_list, fulfilments, info_dict = model(bat, return_pi=True)
            infos = info_dict['inner_info']
        # try:
        gif_path = make_gif(bat, pi, veh_list, fulfilments, infos, opts, step, suffix=f"_val_{i}")
        # except Exception as e:
        #     print(f"Error generating GIF: {e}")
        #     continue
        if hasattr(opts, 'use_wandb') and opts.use_wandb and wandb.run is not None:
            if os.path.exists(gif_path):
                wandb.log({"validation/gif": wandb.Video(gif_path, format="gif")}, step=step)
            
            wandb.log({
                'validation/fulfilment': fulfilments.mean().item() if infos is not None else None,
                'validation/nll_fulfilment': -infos[0].mean().item() if infos is not None else None,
                'validation/alpha_mean': infos[1].mean().item() if infos is not None else None,
                'validation/alpha_std': infos[1].std().item() if infos is not None else None,
                'validation/beta_mean': infos[2].mean().item() if infos is not None else None,
                'validation/beta_std': infos[2].std().item() if infos is not None else None,
            }, step=step)

        if i >= opts.val_gif_limit:
            break

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        # do not need backpropogation
        with torch.no_grad():
            cost, ll, pi, veh_list, fulfilments, info_dict = model(move_to(bat, opts.device), return_pi=True)
            actions = [{'veh': veh_list[i],'node': pi[i],'fulfilment': fulfilments[i]} for i in range(pi.size(0))]
        return (cost.data.cpu(), info_dict, actions)

    # tqdm is a function to show the progress bar
    costs_collection = [
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ]

    actions_collection = []
    for i in range(len(costs_collection)):
        actions_collection += costs_collection[i][2]

    return torch.cat([costs[0] for costs in costs_collection], 0), [infos[1] for infos in costs_collection], actions_collection


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped



def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()
    lr_scheduler.step(epoch)

    # Log learning rate to wandb
    if hasattr(opts, 'use_wandb') and opts.use_wandb and wandb.run is not None:
        wandb.log({
            'training/learning_rate': optimizer.param_groups[0]['lr'],
            'training/epoch': epoch
        }, step=step)

    if not opts.no_tensorboard:  # need tensorboard
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size,
        veh_num=opts.veh_num,
        num_samples=opts.epoch_size,
        distribution=opts.data_distribution,
        # seed=opts.seed + epoch
    ))  # data, baseline (cost of data)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        if opts.use_grpo:
            old_model = copy.deepcopy(model)
            train_grpo_batch(
                model,
                old_model,
                optimizer,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger,
                opts
            )
        else:
            train_batch(
                model,
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger,
                opts
            )
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # Log epoch duration to wandb
    if hasattr(opts, 'use_wandb') and opts.use_wandb and wandb.run is not None:
        wandb.log({
            'training/epoch_duration_seconds': epoch_duration
        }, step=step-1)

    # save results every checkpoint_epoches, saving memory
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        checkpoint_path = os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                # rng_state is the state of random generator
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            # save state of runned model in outputs
            checkpoint_path
        )
        
        # Log model checkpoint to wandb
        if hasattr(opts, 'use_wandb') and opts.use_wandb and wandb.run is not None:
            artifact = wandb.Artifact(
                name=f"model-checkpoint-epoch-{epoch}",
                type="model",
                description=f"Model checkpoint at epoch {epoch}"
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    avg_reward = validate(model, val_dataset, opts, step=step)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)  # data, baseline(cost of data)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
    
    
    # Evaluate proposed model, get costs and log probabilities
    # cost, log_likelihood, log_veh = model(x)  # both [batch_size]
    # cost, log_likelihood = model(x)  # both [batch_size]
    cost, log_likelihood, pi, veh_list, fulfilments, info_dict = model(x, return_pi=True)  # both [batch_size]

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    # reinforce_loss = ((cost - bl_val) * (log_likelihood + log_veh)).mean()
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss
    #print('bl_val', bl_val)

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    if torch.isnan(grad_norms[0][0]):
        print("Gradient norm is NaN, skipping step")
        return
    
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, info_dict['inner_info'], tb_logger, opts)
        if hasattr(opts, 'use_wandb') and opts.use_wandb and wandb.run is not None:
            depot_visits = (pi == 0).sum(-1).float()
            wandb.log({
                'training/depot_visits_mean': depot_visits.mean().item(),
                'training/depot_visits_std': depot_visits.std().item(),
                'training/depot_visits_max': veh_list.max().item(),
            }, step=step)
        
        # Generate GIFs and log them to wandb but only a 10th of the time
        if step % (int(opts.log_step) * 10) == 0:
            try:
                gif_path = make_gif(x, pi, veh_list, fulfilments, None, opts, step)
                if hasattr(opts, 'use_wandb') and opts.use_wandb and wandb.run is not None:
                    if os.path.exists(gif_path):
                        wandb.log({"training/gif": wandb.Video(gif_path, format="gif")}, step=step)
            except Exception as e:
                print(f"Error generating or logging GIF: {e}")


def train_grpo_batch(
    model,
    old_model,
    optimizer,
    epoch,
    batch_id,
    step,
    batch,
    tb_logger,
    opts
):
    x = move_to(batch, opts.device)
    batch_size = x['loc'].shape[0]

    # Expand each sample in the batch to group_size (repeat along new axis)
    # do that for each key in x
    x_group = {key: x[key].unsqueeze(1).expand(-1, opts.grpo_groupsize, *x[key].shape[1:]) for key in x}  # [B, G, ...]
    x_group = {key: x_group[key].reshape(batch_size * opts.grpo_groupsize, *x[key].shape[1:]) for key in x}  # [B*G, ...]

    # Get current policy outputs for all group samples
    cost, log_likelihood, pi, veh_list, fulfilments, info_dict = model(x_group, return_pi=True)  # [B*G]
    cost = cost.view(batch_size, opts.grpo_groupsize)
    log_likelihood = log_likelihood.view(batch_size, opts.grpo_groupsize).clamp(min=-20, max=20)

    # Get old policy log probabilities for all group samples (no grad)
    with torch.no_grad():
        actions = [{'veh': veh_list[:, i],'node': pi[:, i],'fulfilment': fulfilments[:, i]} for i in range(pi.size(1))]
        _, old_log_likelihood, _, _, _, _ = old_model(x_group, actions=actions, return_pi=True)
    old_log_likelihood = old_log_likelihood.view(batch_size, opts.grpo_groupsize).clamp(min=-20, max=20)

    # Group Relative Advantage Estimation
    reward = -cost
    group_mean_reward = reward.mean(dim=1, keepdim=True)  # [B, 1]
    advantage = (reward - group_mean_reward) / (reward.std(dim=1, keepdim=True) + 1e-3)  # [B, G]
    advantage = advantage.detach()  # important!

    # Compute PPO ratio and surrogate loss
    ratio = torch.exp(log_likelihood - old_log_likelihood)  # [B, G]
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - opts.clip_epsilon, 1 + opts.clip_epsilon) * advantage
    ppo_loss = -torch.min(surr1, surr2).mean()

    # Positive unbiased KL divergence (Schulman 2020): ref/cur - log(ref/cur) - 1
    # ref = exp(old_log_likelihood), cur = exp(log_likelihood)
    ref = torch.exp(old_log_likelihood)
    cur = torch.exp(log_likelihood)
    kl_pos = (ref / cur - (old_log_likelihood - log_likelihood) - 1).mean()

    loss = ppo_loss + opts.kl_coef * kl_pos

    optimizer.zero_grad()
    loss.backward()
    grad_norms = clip_grad_norms(optimizer.param_groups)  # no clipping
    # if torch.isnan(grad_norms[0][0]):
    #     print("Gradient norm is NaN, skipping step")
    #     return
    optimizer.step()

    # Logging (minimal, can be expanded)
    if step % int(opts.log_step) == 0:
        log_values(cost.flatten(), grad_norms, epoch, batch_id, step,
                   log_likelihood.flatten(), ppo_loss, 0, None, tb_logger, opts)
        if hasattr(opts, 'use_wandb') and opts.use_wandb and wandb.run is not None:
            wandb.log({
                'training/ppo_loss': ppo_loss.item(),
                'training/kl_pos': kl_pos.item(),
                'training/group_mean_reward': group_mean_reward.mean().item(),
                'training/advantage_mean': advantage.mean().item(),
                'training/advantage_std': advantage.std().item(),
            }, step=step)