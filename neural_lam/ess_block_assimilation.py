import torch
import argparse
from neural_lam.weather_dataset import WeatherDataset
from neural_lam.models.crps import CRPS


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def main(input_args=None):
    """
    Main function for training and evaluating models
    """
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="neural_lam/data_config.yaml",
        help="Path to data config file (default: neural_lam/data_config.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_lam",
        help="Model architecture to train/evaluate (default: graph_lam)",
    )
    parser.add_argument(
        "--subset_ds",
        action="store_true",
        help="Use only a small subset of the dataset, for debugging"
        "(default: false)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        action="store_true",
        help="If optimizer state should be restored with model "
        "(default: false)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )
    parser.add_argument(
        "--num_sanity_steps",
        type=int,
        default=2,
        help="Number of sanity checking validation steps to run before starting"
        " training (default: 2)",
    )

    # Model architecture
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model "
        "(default: multiscale)",
    )
    parser.add_argument(
        "--diffusion_model",
        type=str,
        default="edm",
        help="Model to use in the diffusion model"
        "(default: edm)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Dimensionality of all hidden representations (default: 64)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Dimensionality of latent R.V. at each node (if different than"
        " hidden_dim) (default: None (same as hidden_dim))",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=6,
        help="Number of GNN layers in processor GNN (for prob. model: in "
        "decoder) (default: 6)",
    )
    parser.add_argument(
        "--encoder_processor_layers",
        type=int,
        default=1,
        help="Number of on-mesh GNN layers in encoder GNN (default: 2)",
    )
    parser.add_argument(
        "--prior_processor_layers",
        type=int,
        default=1,
        help="Number of on-mesh GNN layers in prior GNN (default: 2)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) "
        "(default: sum)",
    )
    parser.add_argument(
        "--output_std",
        action="store_true",
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: False (no))",
    )
    parser.add_argument(
        "--shared_grid_embedder",
        action="store_true",  # Default to separate embedders
        help="If the same embedder MLP should be used for interior and boundary"
        " grid nodes. Note that this requires the same dimensionality for "
        "both kinds of grid inputs. (default: False (no))",
    )
    parser.add_argument(
        "--prior_dist",
        type=str,
        default="isotropic",
        help="Structure of Gaussian distribution in prior network output "
        "(isotropic/diagonal) (default: isotropic)",
    )
    parser.add_argument(
        "--learn_prior",
        type=int,
        default=1,
        help="If the prior should be learned as a mapping from previous state "
        "and forcing, otherwise static with mean 0 (default: 1 (yes))",
    )
    parser.add_argument(
        "--vertical_propnets",
        type=int,
        default=0,  # TODO: Change to 1 as it is used in the paper
        help="If PropagationNets should be used for all vertical message "
        "passing (g2m, m2g, up in hierarchy), in deterministic models."
        "(default: 0 (no))",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="heun",
        help="The sampler to use when generating trajectories with a diffusion model"
        "(heun/edm) (default: heun)",
    )

    # Training options
    parser.add_argument(
        "--ar_steps",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss (1-19) "
        "(default: 1)",
    )
    parser.add_argument(
        "--control_only",
        action="store_true",
        help="Train only on control member of ensemble data "
        "(default: False)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="wmse",
        help="Loss function to use, see metric.py (default: wmse)",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=3,
        help="Step length in hours to consider single time step 1-3 "
        "(default: 3)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=1.0,
        help="Beta weighting in front of kl-term in ELBO (default: 1)",
    )
    parser.add_argument(
        "--crps_weight",
        type=float,
        default=0,
        help="Weighting for CRPS term of loss, not computed if = 0. CRPS is "
        "computed based on trajectories sampled using prior distribution. "
        "(default: 0)",
    )
    parser.add_argument(
        "--sample_obs_noise",
        type=int,
        default=0,
        help="If observation noise should be sampled during rollouts (both "
        "training and eval), or just mean prediction used "
        "(default: 0 (no))",
    )
    parser.add_argument(
        "--border_condition",
        action="store_true",
        help="If border condition should be used in diffusion model ",
    )

    parser.add_argument(
        "--pred_residual",
        action="store_true",
        help="If the model should predict residuals instead of absolute values",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for training. (default: 0.01)",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        help="Learning rate scheduler to use, supported (cosine), (default: None)",
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.002,
        help="Sigma min for training. (default: 0.002)",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=88,
        help="Sigma min for training. (default: 88)",
    )
    parser.add_argument(
        "--sigma_coef",
        type=float,
        default=1,
        help="Sigma coefficient for stochatic interpolants (default: 1)",
    )

    # EDM Options
    parser.add_argument(
        "--resample_filter",
        type=list_of_ints,
        default="1,1",
        help="Resample filter for edm model (default: 1,1 or 1,3,3,1)",
    )
    parser.add_argument(
        "--channel_mult",
        type=list_of_ints,
        default="1,2,2,2",
        help="Channel multiplier for edm model (depth and width of UNET) (default: 1,2,2,2)",
    )

    parser.add_argument(
        "--encoder_type",
        type=str,
        default="standard",
        help="Type of encoder to use in edm model (standard/residual/skip)"
        "(default: 'standard')",
    )

    parser.add_argument(
        "--attn_resolutions",
        type=list_of_ints,
        default="1",
        help="Resolutions to apply attention to in edm model (default: '1')",
    )

    parser.add_argument(
        "--noise_embedding",
        type=str,
        default="fourier",
        help="Type of encoder to use in edm model (positional/fourier)"
        "(default: 'fourier')",
    )

    # CRPS Options
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=32,
        help="Dimension of the noise vector z, 32 in FGN (default: 32)",
    )

    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during val/test "
        "(default: 1)",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of ensemble members during evaluation (default: 5)",
    )
    parser.add_argument(
        "--sampler_steps",
        type=int,
        default=20,
        help="Number of sampling steps during inference (default: 20)",
    )
    parser.add_argument(
        "--plot_diffusion_steps",
        action="store_true",
        help="If the diffusion steps should be saved, only one time step is saved",
    )
    parser.add_argument(
        "--noise_aug_prob",
        type=float,
        default=0,
        help="Probability of noise augmentation for training (default: 0)",
    )
    parser.add_argument(
        "--save_output",
        action="store_true",
        help="If the model output should be saved to the output folder (default: False)",
    )
    parser.add_argument(
        "--save_steps",
        action="store_true",
        help="If the model output should be saved to the output folder (default: False)",
    )

    # tEDM Options
    parser.add_argument(
        "--v",
        type=float,  # TODO: Could be tensor with different values for each variable
        default=3.0,  # 3, 5 in the paper
        # NOTE: Heavier tails for lower v, gaussian for v -> ∞
        help="v > 2 parameter for tEDM (default: 3)",
    )

    # Logger Settings
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="neural-lam_prob",
        help="Wandb run project (default: 'neural-lam_prob')",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="Wandb run name (default: '')",
    )
    parser.add_argument(
        "--val_steps_to_log",
        type=list,
        default=[1, 2, 3, 5, 10, 15, 19],
        help="Steps to log val loss for (default: [1, 2, 3, 5, 10, 15, 19])",
    )
    parser.add_argument(
        "--metrics_watch",
        nargs="+",
        default=[],
        help="List of metrics to watch, including any prefix (e.g. val_rmse)",
    )
    parser.add_argument(
        "--var_leads_metrics_watch",
        type=str,
        default="{}",
        help="""JSON string with variable-IDs and lead times to log watched
             metrics (e.g. '{"1": [1, 2], "3": [3, 4]}')""",
    )
    parser.add_argument(
        "--save_output_wandb",
        action="store_true",
        help="If the model output should be saved to wandb (save_output has to be enabled)",
    )

    args = parser.parse_args(input_args)
    args.var_leads_metrics_watch = {
        int(k): v for k, v in json.loads(args.var_leads_metrics_watch).items()
    }

    torch.manual_seed(42)
    standardize = True
    args.pred_length = 5

    dataset = WeatherDataset(
        "meps",
        pred_length=args.pred_length,
        split="test",
        subsample_step=3,
        subset=bool(1),
        control_only=0,
        standardize=standardize,
        border_condition=True
    )

    sample_idx = 0  # torch.randint(len(dataset), size=(1,)).item()
    (init_states, target_states, forcing,
     boundary_forcing) = dataset[sample_idx]


if __name__ == "__main__":
    main()
