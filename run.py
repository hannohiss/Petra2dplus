#!/usr/bin/env python
import datetime
import os
import json
import pprint as pp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
import numpy as np
import wandb

from nets.critic_network import CriticNetwork
from options import get_options
from problems.petra.generator_petra import PetraGeneratorRandom
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from utils import torch_load_cpu, load_problem
import warnings

from src.generator_for_petra2dplus import generator_for_petra2dplus

def run(opts):

    # Initialize wandb if enabled
    if opts.use_wandb:
        wandb.init(
            project=opts.wandb_project,
            entity=opts.wandb_entity,
            name=opts.run_name,
            config=vars(opts),
            # sync_tensorboard=True,
            save_code=True
        )

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        # tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_v{}_c{}".format(opts.problem,opts.veh_num,opts.graph_size), opts.run_name))
        pass

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    ###### Petra
    # ADJUST TO PETRA PROBLEM
    # Load the generator from the data adapter
    vehicle_candidates = [
        {"load": 20000},
        {"load": 35000},
    ] + [{"load": 40000}] * 5
    # get opts.veh_num amount of vehicles
    selected_vehicles = []
    for i in range(opts.veh_num):
        selected_vehicles.append(vehicle_candidates[i % len(vehicle_candidates)])
    opts.vehicles = selected_vehicles

    if opts.use_data_adapter:
        generator, vehicles_ = generator_for_petra2dplus(opts)
        opts.vehicles = vehicles_
    else:
        generator = PetraGeneratorRandom(
            opts,
            num_loc=opts.graph_size,
            vehicles=selected_vehicles,
        )

    # Figure out what's the problem
    if opts.problem == "petra":
        from utils.functions import load_petra_problem
        problem = load_petra_problem(opts.problem, generator=generator)
    else:
        problem = load_problem(opts.problem)
    # problem = HCVRP(opts.graph_size,opts.veh_num,opts.obj)

    # Load data from load_path
    # if u have run the model before, u can continue from resume path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        opts.obj,
        problem,
        n_heads=opts.n_heads,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        opts=opts,
    ).to(opts.device)

    # multi-gpu
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state from trained model
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    def lr_lambda(epoch):
        if epoch < opts.lr_warmup_epochs:
            # Linear warmup
            return float(epoch + 1) / opts.lr_warmup_epochs
        else:
            # Decay after warmup
            return opts.lr_decay ** (epoch - opts.lr_warmup_epochs)

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size, veh_num=opts.veh_num, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    torch.autograd.set_detect_anomaly(True)

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        print("Start training...")
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )

        # Final validation after training
        print("Training completed, running final validation...")
        final_val_result = validate(model, problem.make_dataset(
                size=opts.graph_size,
                veh_num=opts.veh_num,
                num_samples=opts.val_size), 
            opts, 
            step=opts.n_epochs * (opts.epoch_size // opts.batch_size)
        )

        # Log final validation results to wandb
        if opts.use_wandb:
            wandb.log({
                'final_validation/avg_cost': final_val_result.item(),
                'final_validation/epoch': opts.n_epochs
            })
            
            # Log final model as artifact
            final_model_path = os.path.join(opts.save_dir, f'final_model_epoch_{opts.n_epochs}.pt')
            torch.save({
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict(),
                'epoch': opts.n_epochs
            }, final_model_path)
            
            artifact = wandb.Artifact(
                name=f"model-final-{opts.run_name}",
                type="model",
                description=f"Final model after {opts.n_epochs} epochs"
            )
            artifact.add_file(final_model_path)
            wandb.log_artifact(artifact)
            
            # Close wandb
            wandb.finish()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run(get_options())
