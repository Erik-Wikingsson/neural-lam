from typing import Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
import logging
import torch

from neural_lam.models.crps import CRPS
import pytorch_lightning as pl


class ESS(pl.LightningModule):
    """
    Base class for data assimilation methods.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Initialize FGN Model
        self.model = CRPS(args).to(args.device)
        self.model.load_state_dict(torch.load(args.pretrained_model_path,
                                              map_location=args.device)["state_dict"])
        print(f"args.device: {args.device}")

        self.model.eval()

    def log_likelihood_fn(self, z, init_states, forcing, boundary_forcing, obs):
        """
        Compute the log-likelihood of the observations given the forecasted states.
        Args:
            init_states: Initial states, shape (n_ens, state_dim).
            obs: Observations, shape (obs_dim,).
        Returns:
            log_likelihood: Log-likelihood of the observations, shape (n_ens,).
        """
        # [B, noise_dim]
        # prev_state, prev_prev_state, forcing, boundary_forcing
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        with torch.no_grad():
            forecast, _ = self.model.predict_step(
                prev_state, prev_prev_state, forcing, boundary_forcing, z)
            logp = -0.5 * ((self.observation_fn(forecast) -
                           obs) ** 2) / self.args.obs_sigma**2
        return torch.sum(logp, dim=(-2, -1))

    def create_log_likelihood_fn(self, init_states, forcing, boundary_forcing, obs):
        return lambda z: self.log_likelihood_fn(z, init_states, forcing, boundary_forcing, obs)

    def block_log_likelihood_fn(self, z_block, init_states, forcing, boundary_forcing, obs_block):
        """
        Compute the log-likelihood of the observations given the forecasted states.
        Args:
            init_states: Initial states, shape (n_ens, state_dim).
            obs: Observations, shape (obs_dim,).
        Returns:
            log_likelihood: Log-likelihood of the observations, shape (n_ens,).
        """
        z_block = z_block.reshape(obs_block.shape[0], init_states.shape[0], -1)
        logp_sums = []

        for z, obs in zip(z_block, obs_block):
            prev_prev_state = init_states[:, 0]
            prev_state = init_states[:, 1]
            # [B, noise_dim]
            with torch.no_grad():
                forecast, _ = self.model.predict_step(
                    prev_state, prev_prev_state, forcing, boundary_forcing, z)
                init_states = torch.cat(
                    [init_states[:, 1:], forecast.unsqueeze(1)], dim=1)

                if obs is not None:
                    logp = -0.5 * ((self.observation_fn(forecast) -
                                    obs) ** 2) / self.args.obs_sigma**2
                    logp_sums.append(torch.sum(logp, dim=(-2, -1)))
        logp_sum = torch.stack(logp_sums, dim=0).flatten()

        return logp_sum

    def create_block_log_likelihood_fn(self, init_states, forcing, boundary_forcing, obs):
        return lambda z: self.block_log_likelihood_fn(z, init_states, forcing, boundary_forcing, obs)

    def assimilate(self, x_forecast, init_states, forcing, boundary_forcing, obs, obs_mask, obs_fn, debug=False, **kwargs):
        """
        Perform a data assimilation step.
        Args:
            x_forecast: Forecast state, shape (n_ens, state_dim). NOTE: Should be scaled by scalefact.
            init_states: Initial states for the assimilation, shape (window, n_ens, state_dim, X, Y). NOTE: Should be scaled by scalefact.
            obs: Observations, shape (obs_dim,).
            obs_mask: Mask for observations.
            obs_fn: Observation operator function.
            **kwargs: Additional keyword arguments that can be passed to specific implementations.

        Returns:
            x_analysis: Analyzed state after assimilation, shape (n_ens, state_dim). NOTE: Should be scaled by scalefact.
        """
        # Scale and normalize init_states
        init_states_scaled = torch.tensor(init_states, device=self.args.device)

        self.obs_mask = torch.tensor(obs_mask, device=self.args.device)
        obs = torch.tensor(obs, device=self.args.device)
        self.observation_fn = lambda x: obs_fn(
            x[:, self.obs_mask[0, :, 0].bool(), :])
        self.current_log_likelihood_fn = self.create_log_likelihood_fn(
            init_states_scaled, forcing, boundary_forcing, obs)

        self.elliptical_sampler = AsyncEllipticalSliceSampler(
            self.current_log_likelihood_fn,
            max_subiter=self.args.max_subiter,
            logging_interval=1,
        )

        # [n_ens, noise_dim]
        init_z = torch.randn(
            init_states_scaled.shape[0], self.model.model.noise_dim, device=self.args.device)

        if debug:
            analysis_z, log_like_list = self.elliptical_sampler.sample(init_z,
                                                                       num_samples=self.args.num_ess_samples,
                                                                       num_warmup=self.args.num_ess_warmup,
                                                                       return_nfe=False,
                                                                       debug=debug,
                                                                       )
            # [n_ens, num_samples, noise_dim]
            # NOTE: We can take more samples and then use some spread out indexes to get approximately iid samples.
            with torch.no_grad():
                prev_prev_state = init_states[:, 0]
                prev_state = init_states[:, 1]
                analysis, _ = self.model.predict_step(
                    prev_state, prev_prev_state, forcing, boundary_forcing, analysis_z[:, 0, :])  # Take the first sample

            return analysis, log_like_list
        else:
            analysis_z, nfe_per_sample = self.elliptical_sampler.sample(init_z,
                                                                        num_samples=self.args.num_ess_samples,
                                                                        num_warmup=self.args.num_ess_warmup,
                                                                        return_nfe=False,
                                                                        debug=debug,
                                                                        )
            # [n_ens, num_samples, noise_dim]
            # NOTE: We can take more samples and then use some spread out indexes to get approximately iid samples.
            with torch.no_grad():
                prev_prev_state = init_states[:, 0]
                prev_state = init_states[:, 1]
                analysis, _ = self.model.predict_step(
                    prev_state, prev_prev_state, forcing, boundary_forcing, analysis_z[:, 0, :])  # Take the first sample

            return analysis

    def assimilate_block(self, x_forecast, init_states, forcing, boundary_forcing, obs, obs_mask, obs_fn, debug=False, **kwargs):
        """
        Perform a data assimilation step.
        Args:
            x_forecast: Forecast state, shape (n_ens, state_dim). NOTE: Should be scaled by scalefact.
            init_states: Initial states for the assimilation, shape (window, n_ens, state_dim, X, Y). NOTE: Should be scaled by scalefact.
            obs: Observations, shape (n_times, obs_dim,).
            obs_mask: Mask for observations.
            obs_fn: Observation operator function.
            **kwargs: Additional keyword arguments that can be passed to specific implementations.

        Returns:
            x_analysis: Analyzed state after assimilation, shape (n_ens, state_dim). NOTE: Should be scaled by scalefact.
        """
        # Scale and normalize init_states
        init_states_scaled = torch.tensor(init_states, device=self.args.device)

        self.obs_mask = torch.tensor(obs_mask, device=self.args.device)
        self.observation_fn = lambda x: obs_fn(x[:, self.obs_mask[0].bool()])
        self.current_log_likelihood_fn = self.create_block_log_likelihood_fn(
            init_states_scaled, forcing, boundary_forcing, obs)

        self.elliptical_sampler = AsyncEllipticalSliceSampler(
            self.current_log_likelihood_fn,
            max_subiter=self.args.max_subiter,
            logging_interval=1,
        )

        # [n_ens, noise_dim]
        init_z = torch.randn(
            len(obs), init_states_scaled.shape[0], self.model.model.noise_dim, device=self.args.device)

        if debug:
            analysis_z, log_like_list = self.elliptical_sampler.sample(init_z.flatten(0, 1),
                                                                       num_samples=self.args.num_ess_samples,
                                                                       num_warmup=self.args.num_ess_warmup,
                                                                       return_nfe=False,
                                                                       debug=debug,
                                                                       )
            # [n_ens, num_samples, noise_dim]
            # NOTE: We can take more samples and then use some spread out indexes to get approximately iid samples.
            analysis_list = []
            analysis_z = analysis_z.reshape(
                len(obs), init_states_scaled.shape[0], analysis_z.shape[1], analysis_z.shape[2])
            print(f"analysis_z shape: {analysis_z.shape}")
            with torch.no_grad():
                for idx, z in enumerate(analysis_z):
                    prev_prev_state = init_states[:, 0]
                    prev_state = init_states[:, 1]
                    analysis, _ = self.model.predict_step(
                        prev_state, prev_prev_state, forcing, boundary_forcing, z[:, 0, :])  # Take the first sample
                    analysis_list.append(analysis)
                    init_states_scaled = torch.cat(
                        [init_states_scaled[:, 1:], analysis.unsqueeze(1)], dim=1)
            analysis = torch.stack(analysis_list, dim=0)

            return analysis, log_like_list
        else:
            analysis_z, nfe_per_sample = self.elliptical_sampler.sample(init_z.flatten(0, 1),
                                                                        num_samples=self.args.num_ess_samples,
                                                                        num_warmup=self.args.num_ess_warmup,
                                                                        return_nfe=False,
                                                                        debug=debug,
                                                                        )
            analysis_z = analysis_z.reshape(
                len(obs), init_states_scaled.shape[0], analysis_z.shape[1], analysis_z.shape[2])
            # [n_times, n_ens, num_samples, noise_dim]
            # NOTE: We can take more samples and then use some spread out indexes to get approximately iid samples.
            analysis_list = []
            with torch.no_grad():
                for idx, z in enumerate(analysis_z):
                    prev_prev_state = init_states[:, 0]
                    prev_state = init_states[:, 1]
                    analysis, _ = self.model.predict_step(
                        prev_state, prev_prev_state, forcing, boundary_forcing, z[:, 0, :])  # Take the first sample
                    analysis_list.append(analysis)
                    init_states_scaled = torch.cat(
                        [init_states_scaled[:, 1:], analysis.unsqueeze(1)], dim=1)
            analysis = torch.stack(analysis_list, dim=0)

            return analysis


@dataclass
class SampleState:
    x: torch.Tensor
    nu: torch.Tensor
    angle: torch.Tensor
    log_like: torch.Tensor
    log_thresh: torch.Tensor
    theta_min: torch.Tensor
    theta_max: torch.Tensor
    mcmc_iter: torch.Tensor
    sub_iter: torch.Tensor
    nfe: torch.Tensor


class AsyncEllipticalSliceSampler:
    def __init__(
        self,
        log_likelihood_fn: Callable[[torch.Tensor], torch.Tensor],
        max_subiter: int = 100,
        logging_interval: int = 100,
    ):
        self.log_likelihood_fn = log_likelihood_fn
        self.max_subiter = max_subiter
        self.logging_interval = logging_interval

        # setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

    def _get_progress_stats(
        self, state: SampleState, total_steps: int, start_time: float
    ):
        """Compute progress statistics for logging."""
        mcmc_min = state.mcmc_iter.min().item()
        mcmc_max = state.mcmc_iter.max().item()

        # Calculate progress and time estimates
        elapsed_time = time.time() - start_time
        progress = mcmc_min / total_steps  # slowest sample determines progress

        if progress > 0:
            estimated_total_time = elapsed_time / progress
            remaining_time = estimated_total_time - elapsed_time
            eta = datetime.now() + timedelta(seconds=remaining_time)
        else:
            eta = "Unknown"
            remaining_time = float("inf")

        return {
            "mcmc_min": mcmc_min,
            "mcmc_max": mcmc_max,
            "elapsed_time": elapsed_time,
            "eta": eta,
            "progress": progress * 100,  # as percentage
            "remaining_time": remaining_time,
            "log_like": state.log_like.mean().item(),
        }

    def _log_progress(self, stats: dict):
        """Format and log progress information."""
        self.log_like_list.append(stats["log_like"])
        self.logger.info(
            f"Progress: {stats['progress']:.1f}% | "
            f"MCMC steps: {stats['mcmc_min']}-{stats['mcmc_max']} | "
            f"Time elapsed: {timedelta(seconds=int(stats['elapsed_time']))} | "
            f"ETA: {stats['eta']} | "
            f"log_like: {stats['log_like']:.4f}"
        )

    def _initialize_states(
        self,
        x: torch.Tensor,
    ) -> SampleState:
        """Initialize the state for all samples"""
        batch_size = x.shape[0]
        device = x.device

        # auxiliary variable to construct the ellipse
        nu = torch.randn_like(x)

        # sample initial proposal angles
        angle = torch.rand(batch_size, device=device) * 2 * torch.pi

        # Compute current log likelihood
        log_like = self.log_likelihood_fn(x)

        # Initialize log likelihood threshold
        print(f"log_like: {log_like.shape}")
        print(f"batch_size: {batch_size}")
        log_thresh = log_like + \
            torch.log(torch.rand(batch_size, device=device))

        # Initialize bracket
        theta_min = angle - 2 * torch.pi
        theta_max = angle

        # Initialize MCMC iteration and subiteration
        mcmc_iter = torch.zeros(batch_size, dtype=torch.long, device=device)
        sub_iter = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Intialize nfe counter
        nfe = torch.ones(batch_size, dtype=torch.long, device=device)

        return SampleState(
            x=x,
            nu=nu,
            angle=angle,
            log_like=log_like,
            log_thresh=log_thresh,
            theta_min=theta_min,
            theta_max=theta_max,
            mcmc_iter=mcmc_iter,
            sub_iter=sub_iter,
            nfe=nfe,
        )

    def _proposal(self, state: SampleState, active_mask: torch.Tensor) -> torch.Tensor:
        active_mask = active_mask.bool()
        cos_angle = torch.cos(state.angle)
        sin_angle = torch.sin(state.angle)

        new_shape = (-1,) + (1,) * (state.x.dim() - 1)
        active_mask = active_mask.reshape(new_shape)
        cos_angle = cos_angle.reshape(new_shape).expand_as(state.x)
        sin_angle = sin_angle.reshape(new_shape).expand_as(state.x)

        proposal = state.x * cos_angle + state.nu * sin_angle
        return torch.where(active_mask, proposal, state.x)

    def _update_state_for_new_mcmc_step(
        self, state: SampleState, accept_mask: torch.Tensor
    ):
        device = state.x.device
        accept_mask = accept_mask.bool()

        # new auxiliary variable for the ellipse
        state.nu = torch.where(
            accept_mask.reshape(-1, *([1] * (state.nu.dim() - 1))),
            torch.randn_like(state.x),
            state.nu,
        )

        # new proposal angle
        state.angle = torch.where(
            accept_mask,
            torch.rand_like(state.angle, device=device) * 2 * torch.pi,
            state.angle,
        )

        # new log likelihood threshold
        state.log_thresh = torch.where(
            accept_mask,
            state.log_like +
            torch.log(torch.rand_like(state.log_like, device=device)),
            state.log_thresh,
        )

        # new bracket
        state.theta_min = torch.where(
            accept_mask,
            state.angle - 2 * torch.pi,
            state.theta_min,
        )
        state.theta_max = torch.where(
            accept_mask,
            state.angle,
            state.theta_max,
        )

        # update MCMC iteration and subiteration
        state.mcmc_iter = torch.where(
            accept_mask,
            state.mcmc_iter + 1,
            state.mcmc_iter,
        )
        state.sub_iter = torch.where(
            accept_mask,
            torch.zeros_like(state.sub_iter, device=device),
            state.sub_iter,
        )

    def _update_state_for_new_subiter(
        self, state: SampleState, reject_mask: torch.Tensor
    ):
        device = state.x.device
        reject_mask = reject_mask.bool()

        # update the bracket
        state.theta_min = torch.where(
            reject_mask & (state.angle < 0),
            state.angle,
            state.theta_min,
        )
        state.theta_max = torch.where(
            reject_mask & (state.angle >= 0),
            state.angle,
            state.theta_max,
        )

        # sample new proposal angle
        state.angle = torch.where(
            reject_mask,
            state.theta_min
            + torch.rand_like(state.angle, device=device)
            * (state.theta_max - state.theta_min),
            state.angle,
        )

        # update subiteration
        state.sub_iter = torch.where(
            reject_mask,
            state.sub_iter + 1,
            state.sub_iter,
        )

        # handle the case where the subiteration exceeds the maximum
        max_subiter_mask = reject_mask & (state.sub_iter >= self.max_subiter)
        if max_subiter_mask.any():
            self._update_state_for_new_mcmc_step(state, max_subiter_mask)

    @torch.no_grad()
    def sample(
        self,
        init_x: torch.Tensor,
        num_samples: int,
        num_warmup: int = 0,
        return_nfe: bool = False,
        debug: bool = False,
    ):
        batch_size = init_x.shape[0]
        total_steps = num_samples + num_warmup

        # intialize timing and iteration tracking
        start_time = time.time()
        iteration_counter = 0

        # intialize samples as init_x with extra dimension for num_samples (1)
        samples = (
            init_x.unsqueeze(1)
            .expand(-1, num_samples, *([-1] * (init_x.dim() - 1)))
            .clone()
        )
        # initialize the state
        state = self._initialize_states(init_x)

        # log the initial state
        self.logger.info(
            f"Starting sampling with batch size {batch_size}, "
            f"total steps {total_steps} (warmup: {num_warmup})"
        )

        # mask for active samples
        active_mask = state.mcmc_iter < total_steps
        self.log_like_list = []

        # iterate until all samples in the batch have passed the MCMC steps
        while active_mask.any():
            iteration_counter += 1

            # log progress
            if iteration_counter % self.logging_interval == 0:
                stats = self._get_progress_stats(
                    state, total_steps, start_time)

                self._log_progress(stats)

            # update the active mask
            active_mask = state.mcmc_iter < total_steps

            # compute the proposal
            x_prop = self._proposal(state, active_mask)
            # print(f"Generated proposal samples: {x_prop.shape}")
            # print(
            #     f"Proposal sample mean: {x_prop.mean().item()}, std: {x_prop.std().item()}")
            # print(
            #     f"Current sample mean: {state.x.mean().item()}, std: {state.x.std().item()}")
            # print(
            #     f"Difference mean: {(x_prop - state.x).mean().item()}, std: {(x_prop - state.x).std().item()}")
            # print(
            #     f"Difference init mean: {(x_prop - init_x).mean().item()}, std: {(x_prop - init_x).std().item()}")

            # compute the log likelihood of the proposal
            log_like_prop = self.log_likelihood_fn(x_prop)

            # increment NFE counter for active samples
            state.nfe = torch.where(
                active_mask,
                state.nfe + 1,
                state.nfe,
            )

            # accept/reject the proposal
            accept_mask = (log_like_prop > state.log_thresh) & active_mask
            reject_mask = (~accept_mask) & active_mask

            # print(f"Iteration {iteration_counter}: "
            #       f"Active samples: {active_mask.sum().item()}, "
            #       f"Rejected samples: {reject_mask.sum().item()}, "
            #       f"Log likelihood: {log_like_prop.mean().item()}")

            if accept_mask.any():
                # update the position and log likelihood
                state.x = torch.where(
                    accept_mask.reshape(-1, *([1] * (state.x.dim() - 1))),
                    x_prop,
                    state.x,
                )
                state.log_like = torch.where(
                    accept_mask, log_like_prop, state.log_like)

                # store the samples
                is_sample = (state.mcmc_iter >= num_warmup) & accept_mask
                if is_sample.any():
                    sample_indices = state.mcmc_iter[is_sample] - num_warmup
                    batch_indices = is_sample.nonzero().squeeze(-1)
                    samples[batch_indices,
                            sample_indices] = state.x[batch_indices]

                # initialize the new MCMC step
                self._update_state_for_new_mcmc_step(state, accept_mask)

            # handle the rejected samples
            if reject_mask.any():
                self._update_state_for_new_subiter(state, reject_mask)

        # log the final state
        final_stats = self._get_progress_stats(state, total_steps, start_time)
        self.logger.info("Sampling completed!")
        self._log_progress(final_stats)
        print(f"Debug: {debug}")

        if return_nfe:
            return samples, state.nfe
        elif debug:
            print(f"Log likelihood list length: {len(self.log_like_list)}")
            return samples, self.log_like_list
        else:
            return samples, None
