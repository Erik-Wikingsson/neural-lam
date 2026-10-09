# Third-party
import torch
import wandb
import matplotlib.pyplot as plt

# First-party
from neural_lam import constants, metrics, vis
from neural_lam.models.ar_prob_model import ARProbModel
from neural_lam.models.edm_networks_2 import SongUNet


class CRPS(ARProbModel):
    """
    A crps based probabilistic auto-regressive weather forecasting model
    """

    def __init__(self, args):
        super().__init__(args)

    # TODO: Define backbone model architecture here
        self.model = SongUNet(
            img_resolution=torch.as_tensor(constants.FULL_GRID_SHAPE),
            in_channels=self.grid_dim,
            in_channels_boundary=self.boundary_dim,
            out_channels=self.grid_output_dim,
            embedding_type="linear",
            resample_filter=args.resample_filter,
            channel_mult=args.channel_mult,
            encoder_type=args.encoder_type,
            attn_resolutions=args.attn_resolutions,
            noise_dim=args.noise_dim,
        )

    def predict_step(self, prev_state, prev_prev_state, forcing, boundary_forcing, z=None):
        """
        Predict weather state one time step ahead
        X_{t-1}, X_t -> X_t+1

        prev_state: (B, N_grid, d_state), weather state X_t at time t
        prev_prev_state: (B, N_grid, d_state), weather state X_{t-1} at time t-1
        batch_static_features: (B, N_grid, batch_static_feature_dim), static forcing
        forcing: (B, N_grid, forcing_dim), dynamic forcing

        Returns:
        next_state: (B, N_grid, d_state), predicted weather state X_{t+1} at time t+1
        pred_std: None or (B, N_grid, d_state), predicted standard-deviations
                    (pred_std can be ignored by just returning None)
        """

        x = torch.cat((prev_state, prev_prev_state, forcing),
                      dim=-1)  # (B, N_grid, d_input)
        if z is None:
            z = torch.randn(
                prev_state.shape[0], self.model.noise_dim, device=prev_state.device)
        next_state = self.model(x, z, boundary_forcing=boundary_forcing)

        if self.pred_residual:
            next_state = (next_state * self.step_diff_std[constants.USED_PARAMS]) + \
                self.step_diff_mean[constants.USED_PARAMS]  # Unormalize residual
            next_state = prev_state + next_state

        return next_state, None

    def training_step(self, batch):
        """
        Train on single batch
        """
        (init_states, target_states, forcing, boundary_forcing) = batch

        # Sample trajectories using prior
        pred_traj_means, pred_traj_stds = self.sample_trajectories(
            init_states,
            forcing,
            boundary_forcing,
            2,  # NOTE: Always sample 2 trajectories? They use N=2 in FGN
        )
        # (B, S=2, pred_steps, num_grid_nodes, d_f), always 2 samples

        # Compute CRPS, TODO: Should we include some weighting of the variables?
        crps_estimate = metrics.crps_ens(  # TODO: Should we use this CRPS version?
            pred_traj_means,
            target_states,
        )  # (B, pred_steps)
        loss = torch.mean(crps_estimate)

        log_dict = {"train_loss": loss}

        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """

        (init_states, target_states, forcing, boundary_forcing) = batch

        # Sample trajectories using prior
        pred_traj_means, pred_traj_stds = self.sample_trajectories(
            init_states,
            forcing,
            boundary_forcing,
            2,  # NOTE: Always sample 2 trajectories? They use N=2 in FGN
        )
        # (B, S=2, pred_steps, num_grid_nodes, d_f), always 2 samples

        # Compute CRPS, TODO: Should we include some weighting of the variables?
        loss = metrics.crps_ens(  # TODO: Should we use this CRPS version?
            pred_traj_means,
            target_states,
            pred_traj_stds,
            # mask=self.interior_mask_bool,
        )  # (B, pred_steps)

        time_step_loss = torch.mean(loss, dim=0)  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            # ONLY LOGGING FOR 1 STEP since logging diffusion steps for all steps is too much and not that informative
            for step in constants.VAL_STEP_LOG_ERRORS
        }
        val_log_dict["val_mean_loss"] = mean_loss
        self.log_dict(
            val_log_dict, on_step=False, on_epoch=True, sync_dist=True
        )

        # (B, pred_steps, num_grid_nodes, d_f), only use first sample for deterministic metrics
        prediction = pred_traj_means[:, 0, :, :, :]

        # Store MSEs
        entry_mses = metrics.mse(
            prediction,
            target_states,
            None,
            # mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.val_metrics["mse"].append(entry_mses)

        # Get Probabilistic Metrics
        (
            trajectories,
            traj_stds,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        ) = self.ensemble_common_step(batch)

        self.val_metrics["spread_squared"].append(spread_squared_batch)
        self.val_metrics["ens_mse"].append(ens_mse_batch)

        # Compute additional ensemble metrics
        ens_mean = torch.mean(
            trajectories, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_std = torch.std(trajectories, dim=1)
        # (B, pred_steps, num_grid_nodes, d_f)

        # Compute MAE for ensemble mean + ensemble CRPS
        ens_maes = metrics.mae(
            ens_mean,
            target_states,
            ens_std,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.val_metrics["ens_mae"].append(ens_maes)
        crps_batch = metrics.crps_ens(
            trajectories,
            target_states,
            None,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.val_metrics["crps_ens"].append(crps_batch)

        # Plot example predictions (on rank 0 only)
        if self.trainer.is_global_zero and batch_idx == 0:
            n_examples = 1  # Number of examples to plot
            border = boundary_forcing[..., :len(constants.USED_PARAMS)]

            # Rescale to original data scale
            traj_rescaled = trajectories * \
                self.data_std[constants.USED_PARAMS] + \
                self.data_mean[constants.USED_PARAMS]
            target_rescaled = target_states * \
                self.data_std[constants.USED_PARAMS] + \
                self.data_mean[constants.USED_PARAMS]
            border_rescaled = border * \
                self.data_std[constants.USED_PARAMS] + \
                self.data_mean[constants.USED_PARAMS]
            # Compute mean and std of ensemble
            ens_mean = torch.mean(
                traj_rescaled, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
            ens_std = torch.std(
                traj_rescaled, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)

            # Iterate over the examples
            for traj_slice, target_slice, border_slice, ens_mean_slice, ens_std_slice in zip(
                traj_rescaled[:n_examples],
                target_rescaled[:n_examples],
                border_rescaled[:n_examples, ..., :len(constants.USED_PARAMS)],
                ens_mean[:n_examples],
                ens_std[:n_examples],
            ):
                # traj_slice is (S, pred_steps, num_grid_nodes, d_f)
                # others are (pred_steps, num_grid_nodes, d_f)

                # Note: min and max values can not be in ensemble mean
                var_vmin = (
                    torch.minimum(
                        traj_slice.flatten(0, 2).min(dim=0)[0],
                        target_slice.flatten(0, 1).min(dim=0)[0],
                    )
                    .cpu()
                    .numpy()
                )  # (d_f,)
                var_vmax = (
                    torch.maximum(
                        traj_slice.flatten(0, 2).max(dim=0)[0],
                        target_slice.flatten(0, 1).max(dim=0)[0],
                    )
                    .cpu()
                    .numpy()
                )  # (d_f,)
                var_vranges = list(zip(var_vmin, var_vmax))

                # Iterate over prediction horizon time steps
                for t_i, (samples_t, target_t, border_t, ens_mean_t, ens_std_t) in enumerate(
                    zip(
                        traj_slice.transpose(0, 1),
                        # (pred_steps, S, num_grid_nodes, d_f)
                        target_slice,
                        border_slice,
                        ens_mean_slice,
                        ens_std_slice,
                    ),
                    start=1,
                ):
                    if t_i in constants.VAL_PLOT_STEPS:
                        time_title_part = f"t={t_i} ({self.step_length*t_i} h)"
                        # Create one figure per variable at this time step
                        var_figs = [
                            vis.plot_ensemble_prediction(
                                samples_t[:, :, var_i],
                                target_t[:, var_i],
                                border_t[:, var_i],
                                ens_mean_t[:, var_i],
                                ens_std_t[:, var_i],
                                self.interior_mask,
                                title=f"{var_name} ({var_unit}), {time_title_part}",
                                vrange=var_vrange,
                            )
                            for var_i, (var_name, var_unit, var_vrange) in enumerate(
                                zip(
                                    constants.PARAM_NAMES_SHORT[constants.USED_PARAMS],
                                    constants.PARAM_UNITS[constants.USED_PARAMS],
                                    var_vranges,
                                )
                            )
                        ]

                        example_title = f"example_{self.plotted_examples}_step_{t_i}"
                        wandb.log(
                            {
                                f"{var_name}_{example_title}": wandb.Image(fig)
                                for var_name, fig in zip(
                                    constants.PARAM_NAMES_SHORT[constants.USED_PARAMS], var_figs
                                )
                            }
                        )
                        plt.close(
                            "all"
                        )  # Close all figs for this time step, saves memory
