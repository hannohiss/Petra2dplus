import os
import imageio
from matplotlib.patches import Patch
import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colormaps


standard_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']


# Create a dedicated directory for GIFs
def ensure_gif_dir(opts):
    gif_dir = os.path.join(os.getcwd(), opts.save_dir, "gif_outputs")
    os.makedirs(gif_dir, exist_ok=True)
    return gif_dir


def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, infos, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Create a dictionary of metrics to log
    metrics = {
        'training/avg_cost': avg_cost,
        'training/actor_loss': reinforce_loss.item(),
        'training/nll': -log_likelihood.mean().item(),
        'training/grad_norm': grad_norms[0],
        'training/grad_norm_clipped': grad_norms_clipped[0],
        # first is ll of fulfilment
        'training/nll_fulfilment': -infos[0].mean().item() if infos is not None else None,
        'training/alpha_mean': infos[1].mean().item() if infos is not None else None,
        'training/alpha_std': infos[1].std().item() if infos is not None else None,
        'training/beta_mean': infos[2].mean().item() if infos is not None else None,
        'training/beta_std': infos[2].std().item() if infos is not None else None,
    }
    
    if opts.baseline == 'critic':
        metrics.update({
            'training/critic_loss': bl_loss.item(),
            'training/critic_grad_norm': grad_norms[1],
            'training/critic_grad_norm_clipped': grad_norms_clipped[1],
        })

    # Log to wandb if enabled
    if hasattr(opts, 'use_wandb') and opts.use_wandb and wandb.run is not None:
        wandb.log(metrics, step=step)

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)


def make_gif(input, pi, veh_list, fulfilments, infos, opts, step, suffix=""):
    # Log gifs to wandb if enabled
    # if hasattr(opts, 'use_wandb') and opts.use_wandb and wandb.run is not None:
    drops = infos[3] if infos is not None else None
    for i in range(input['loc'].size()[0]):
        policy = pi[i]
        vehicle = veh_list[i]
        fulfilment = fulfilments[i]
        locs = input["loc"][i]
        demands = input["demand"][i]
        node_critical_time = input["node_critical_time"][i]
        node_consumption_rate = input["node_consumption_rate"][i]
        depot = input["depot"][i]
        drop = drops[i] if drops is not None else None

        # Create a gif of the vehicle's route
        gif_path=render(
            policy.cpu().numpy(),
            vehicle.cpu().numpy(),
            fulfilment.cpu().numpy(),
            locs.cpu().numpy(),
            demands.cpu().numpy(),
            node_critical_time.cpu().numpy(),
            node_consumption_rate.cpu().numpy(),
            depot.cpu().numpy(),
            drop,
            step,
            opts,
            suffix=suffix,
        )
        return gif_path


def render(
    policy,
    vehicle,
    fulfilment,
    locs,
    demands,
    node_critical_time,
    node_consumption_rate,
    depot,
    drop,
    step,
    opts,
    suffix="",
):
    """Render the PETRA environment encoder rollout."""
    # Create frames directory for temporary frame images
    frames_dir = os.path.join(os.getcwd(), "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Create GIF output directory
    gif_dir = ensure_gif_dir(opts)

    # _, ax = plt.subplots(dpi=500, figsize=(12, 12))
    _, ax = plt.subplots(dpi=100, figsize=(12, 12))

    legend_elements = []

    ### Subtitle for nodes
    subtitle1 = Patch(facecolor='none', edgecolor='white', label='Nodes:')  # Section title (will look like a blank box)
    legend_elements.append(subtitle1)

    ### Plot depot
    depot_idx = 0
    ax.scatter(
        locs[depot_idx, 0],
        locs[depot_idx, 1],
        edgecolors="black",
        facecolors="white",
        s=200,
        linewidths=2,
        marker="s",
        label="Depot",
        zorder=100,
    )

    scale_demands = demands.max()
    ax.scatter(
        -1,
        -1,
        edgecolors="green",
        facecolors="green",
        s=10,
        linewidths=2,
        marker="o",
        alpha=1,
        zorder=50,
        label="Low Demand",
    )
    ax.scatter(
        -1,
        -1,
        edgecolors="green",
        facecolors="green",
        s=50,
        linewidths=2,
        marker="o",
        alpha=1,
        zorder=50,
        label="Medium Demand",
    )
    ax.scatter(
        -1,
        -1,
        edgecolors="green",
        facecolors="green",
        s=100,
        linewidths=2,
        marker="o",
        alpha=1,
        zorder=50,
        label="High Demand",
    )
    # where some data has already been plotted to ax
    handles, labels = ax.get_legend_handles_labels()
    remove_handles = len(handles)
    legend_elements.extend(handles)

    ### Subtitle for Vehicles
    subtitle2 = Patch(facecolor='none', edgecolor='white', label='Vehicles:')  # Section title
    legend_elements.append(subtitle2)

    # Plot fuel stations
    for node_idx in range(1, locs.shape[0]):
        loc = locs[node_idx]
        # Plot station
        ax.scatter(
            loc[0],
            loc[1],
            edgecolors="green",
            facecolors="green",
            s=100 * (demands[node_idx] / scale_demands),
            linewidths=2,
            marker="o",
            alpha=1,
            zorder=50,
        )
        # annotate node_critical_time
        ax.annotate(
            f"TTR {node_critical_time[node_idx]:.0f}d",
            xy=(loc[0], loc[1]),
            xytext=(loc[0] + 0.02, loc[1] + 0.02),
            fontsize=8,
            color="black",
        )
    
    # Create color map for different vehicles
    num_vehicles = len(opts.vehicles)
    color_list = [standard_colors[i % len(standard_colors)] for i in range(num_vehicles)]

    # add vehicles with their colors to the legend
    for i in range(num_vehicles):
        ax.scatter(
            -1,
            -1,
            edgecolors=color_list[i],
            facecolors=color_list[i],
            s=200,
            linewidths=2,
            marker="_",
            label=f"Vehicle {i} - {opts.vehicles[i]['load']} l",
        )

    ax = draw_pie(
        [0.6, 0.4],
        -1,
        -1,
        size=100,
        ax=ax,
        alpha=0.5,
        label="%-Dropped",
    )

    handles, labels = ax.get_legend_handles_labels()
    legend_elements.extend(handles[remove_handles:])

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # Add legend
    ax.legend(handles=legend_elements, loc='upper left', title="Legend")
    # ax.legend(loc='upper left', title="Legend")

    # Remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    veh_locs = torch.Tensor(locs[0]).unsqueeze(0).expand(num_vehicles, 2).clone().cpu().numpy()

    # Plot routes if actions are provided
    images = []
    if policy is not None and vehicle is not None and fulfilment is not None:
        # Plot routes for each vehicle
        for i, (p,v,f) in enumerate(zip(policy, vehicle, fulfilment)):
            # make title with step
            if drop is not None:
                f = drop[i]/opts.vehicles[v]['load']
                ax.set_title(f"Step {i} - Dropped {f*100:.2f}%", fontsize=16)
            else:
                ax.set_title(f"Step {i} - Fulfilmen {f*100:.2f}%", fontsize=16)


            vehicle_color = color_list[v]
            from_loc = veh_locs[v]
            to_loc = locs[p]
            # Draw arrow
            ax.annotate(
                "",
                xy=(to_loc[0], to_loc[1]),
                xytext=(from_loc[0], from_loc[1]),
                arrowprops=dict(
                    arrowstyle="->", color=vehicle_color, lw=1.5, alpha=0.6
                ),
                zorder=25,
            )
            # update vehicle location
            veh_locs[v] = to_loc

            # Draw pie chart for fulfilment
            ax = draw_pie(
                [f, 1 - f],
                to_loc[0] + 0.01,
                to_loc[1] + 0.01,
                size=100,
                ax=ax,
                alpha=0.5,
            )

            filename = os.path.join(frames_dir, f"frame_{i:02d}.png")
            plt.savefig(filename)
            images.append(imageio.imread(filename))

    # ADDITIONAL SAVE
    text = "GRPO" if opts.use_grpo else "Rollout"
    ax.set_title(f"Validation Rollout - {text}", fontsize=16)
    filename = os.path.join(gif_dir, f"validation_rollout_{step:04d}_{suffix}.png")
    plt.savefig(filename)

    # Create a unique gif filename with absolute path
    gif_filename = f"arrows_{step:04d}_{suffix}.gif"
    gif_path = os.path.join(gif_dir, gif_filename)
    
    # Save the gif to the dedicated directory
    imageio.mimsave(gif_path, images, fps=2)
    
    # clean up temporary frame images
    for filename in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, filename))
    
    plt.close()  # Close the figure to prevent memory leaks

    # return the absolute path to the gif
    return gif_path


def draw_pie(
        dist, 
        xpos, 
        ypos, 
        size,
        ax,
        alpha=0.5,
        label=None,
    ):
    assert ax is not None, "ax must be provided"

    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum/ cumsum[-1]
    pie = [0] + cumsum.tolist()

    colors = standard_colors[:len(dist)]

    if len(dist) == 2:
        alphas = [alpha, 0]
    else: 
        alphas = [alpha] * len(dist)

    for idx, (r1, r2) in enumerate(zip(pie[:-1], pie[1:])):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()

        xy = np.column_stack([x, y])

        ax.scatter([xpos], [ypos], marker=xy, s=size, color=colors[idx], zorder=150, alpha=alphas[idx], label=label if idx == 0 else None)

    return ax
