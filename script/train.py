import argparse
import gc
import importlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

_trainer_mod = importlib.import_module("src.trainer")
Trainer = _trainer_mod.Trainer
TrainerConfig = _trainer_mod.TrainerConfig

_dist_mod = importlib.import_module("src.utils.distributed")
cleanup_distributed = _dist_mod.cleanup_distributed
is_main_process = _dist_mod.is_main_process
log0 = _dist_mod.log0
setup_distributed = _dist_mod.setup_distributed


class NullTrialReporter:
    def report_intermediate(self, metrics, epoch: int, score: float) -> None:
        _ = metrics
        _ = epoch
        _ = score

    def report_final(self, summary) -> None:
        _ = summary


class NNITrialReporter:
    def __init__(self, nni_module) -> None:
        self.nni = nni_module

    @staticmethod
    def _metric_key(name: str) -> str:
        return str(name).replace("@", "At").replace(" ", "_")

    def report_intermediate(self, metrics, epoch: int, score: float) -> None:
        payload = {
            "default": float(score),
            "epoch": int(epoch),
        }
        for key, value in dict(metrics).items():
            payload[self._metric_key(key)] = float(value)
        self.nni.report_intermediate_result(payload)

    def report_final(self, summary) -> None:
        best_val_metrics = dict(summary.get("best_val_metrics", {}))
        test_metrics = dict(summary.get("test_metrics", {}))
        payload = {
            "default": float(summary.get("best_val_score", 0.0)),
        }
        for prefix, metrics in (
            ("best_val", best_val_metrics),
            ("test", test_metrics),
        ):
            for key, value in metrics.items():
                payload[f"{prefix}_{self._metric_key(key)}"] = float(value)
        self.nni.report_final_result(payload)


def _metric_mean(summaries, key: str, section: str) -> float:
    values = [float(summary.get(section, {}).get(key, 0.0)) for summary in summaries]
    return float(np.mean(values)) if values else 0.0


def aggregate_run_summaries(
    summaries,
    primary_metric: str,
) -> dict:
    primary_metric = str(primary_metric).lower()
    metric_key_by_name = {
        "auc": "AUC",
        "aupr": "AUPR",
        "mrr": "MRR",
        "map": "MAP",
        "hits@1": "Hits@1",
        "hits@3": "Hits@3",
        "hits@10": "Hits@10",
        "f1": "F1",
        "maxf1": "MaxF1",
        "mcc": "MCC",
    }
    primary_key = metric_key_by_name.get(primary_metric, "AUPR")

    def collect_metrics(section: str) -> dict:
        keys = []
        for summary in summaries:
            for key in dict(summary.get(section, {})).keys():
                if key not in keys:
                    keys.append(key)
        preferred = ["Loss", "AUC", "AUPR"]
        ordered = [key for key in preferred if key in keys] + [
            key for key in keys if key not in preferred
        ]
        return {key: _metric_mean(summaries, key, section) for key in ordered}

    best_val_metrics = collect_metrics("best_val_metrics")
    test_metrics = collect_metrics("test_metrics")
    best_val_score = float(best_val_metrics.get(primary_key, 0.0))
    return {
        "best_val_score": float(best_val_score),
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }


def get_args():
    parser = argparse.ArgumentParser(
        description="train script for the simplified SRR model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train.yaml",
        help="Path to the config file (YAML format)",
    )
    parser.add_argument("--data_dir", type=str, help="path to the data directory")
    parser.add_argument(
        "--neg_sampler_path",
        type=str,
        help="optional path to neg_sampler.pkl for online negative resampling",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        help="path to save the best model",
    )
    parser.add_argument("--hidden_channels", type=int, help="hidden channels")
    parser.add_argument("--out_channels", type=int, help="output channels")
    parser.add_argument(
        "--num_srr_layers",
        type=int,
        help="number of SRR blocks",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        help="number of attention heads shared by SRR and global router",
    )
    parser.add_argument("--dropout", type=float, help="dropout rate")
    parser.add_argument(
        "--use_global_router",
        action="store_true",
        default=True,
        help="enable the global router branch",
    )
    parser.add_argument(
        "--disable_global_router",
        action="store_true",
        help="disable the global router branch",
    )
    parser.add_argument(
        "--mechanism_dropout",
        type=float,
        default=0.1,
        help="dropout rate inside the router branch",
    )
    parser.add_argument(
        "--decoder_type",
        type=str,
        choices=["complex", "bilinear", "pair_mlp"],
        help="decoder family",
    )
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--weight_decay", type=float, help="weight decay for AdamW")
    parser.add_argument("--epochs", type=int, help="maximum number of epochs")
    parser.add_argument("--warmup_epochs", type=int, help="number of warmup epochs")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        help="early stopping patience in eval steps",
    )
    parser.add_argument("--eval_interval", type=int, help="evaluate every N epochs")
    parser.add_argument(
        "--log_interval", type=int, help="log training loss every N epochs"
    )
    parser.add_argument("--grad_clip_norm", type=float, help="gradient clipping norm")
    parser.add_argument(
        "--source_batch_size",
        type=int,
        help="number of source nodes per grouped InfoNCE batch",
    )
    parser.add_argument(
        "--train_neg_per_pos",
        type=int,
        help="target number of online sampled negatives per positive during training",
    )
    parser.add_argument(
        "--online_negative_sampling",
        action="store_true",
        help="enable online negative resampling from neg_sampler.pkl",
    )
    parser.add_argument(
        "--disable_online_negative_sampling",
        action="store_true",
        help="disable online negative resampling and fall back to offline negatives",
    )
    parser.add_argument(
        "--use_contraindication_negatives",
        action="store_true",
        help="add train-time contraindication edges as explicit negatives",
    )
    parser.add_argument(
        "--disable_contraindication_negatives",
        action="store_true",
        help="disable contraindication negatives in grouped training batches",
    )
    parser.add_argument(
        "--negative_curriculum_epochs",
        type=int,
        help="epochs used to ramp candidate difficulty toward the target setting",
    )
    parser.add_argument(
        "--negative_start_factor",
        type=float,
        help="start ratio for train_neg_per_pos during curriculum warmup",
    )
    parser.add_argument(
        "--hard_negative_ratio_start",
        type=float,
        help="starting hard-negative ratio before curriculum ramps to sampler target",
    )
    parser.add_argument(
        "--min_learning_rate_ratio",
        type=float,
        help="minimum learning-rate ratio reached by cosine decay after warmup",
    )
    parser.add_argument(
        "--monitor_gradients",
        action="store_true",
        help="whether to log gradient norms",
    )
    parser.add_argument(
        "--infonce_temperature",
        type=float,
        help="InfoNCE softmax temperature",
    )
    parser.add_argument(
        "--infonce_loss_mode",
        type=str,
        choices=["pooled", "per_positive"],
        help="grouped InfoNCE reduction over multiple positives for the same source",
    )
    parser.add_argument(
        "--aux_bce_weight",
        type=float,
        help="weight of the auxiliary class-balanced BCE loss added to InfoNCE",
    )
    parser.add_argument(
        "--aux_focal_enabled",
        action="store_true",
        help="enable auxiliary focal loss on grouped candidates",
    )
    parser.add_argument(
        "--disable_aux_focal",
        action="store_true",
        help="disable auxiliary focal loss",
    )
    parser.add_argument(
        "--aux_focal_alpha",
        type=float,
        help="alpha parameter for auxiliary focal loss",
    )
    parser.add_argument(
        "--aux_focal_gamma",
        type=float,
        help="gamma parameter for auxiliary focal loss",
    )
    parser.add_argument(
        "--aux_focal_weight",
        type=float,
        help="weight of auxiliary focal loss added to InfoNCE",
    )
    parser.add_argument(
        "--edge_dropout_p",
        type=float,
        help="drop probability applied to message-passing edges during training",
    )
    parser.add_argument(
        "--primary_metric",
        type=str,
        choices=["auc", "aupr", "mrr", "map", "hits@1", "hits@3", "hits@10", "f1", "maxf1", "mcc"],
        help="checkpoint selection metric",
    )
    parser.add_argument(
        "--ranking_group_axis",
        type=str,
        choices=["src", "source", "chemical", "0", "dst", "target", "disease", "1"],
        help="grouping axis for ranking metrics; use dst/disease for cold-start retrieval",
    )
    parser.add_argument(
        "--training_group_axis",
        type=str,
        choices=["src", "source", "chemical", "0", "dst", "target", "disease", "1"],
        help="grouping axis for grouped InfoNCE training",
    )
    parser.add_argument(
        "--full_ranking_candidate_batch_size",
        type=int,
        help="number of full-candidate ranking edges scored per decoder chunk",
    )
    parser.add_argument("--gpu_id", type=int, help="GPU ID to use, -1 for CPU")
    parser.add_argument(
        "--nni",
        action="store_true",
        help="enable NNI trial parameter injection and result reporting",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="enable DDP multi-process training",
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="distributed backend",
    )
    parser.add_argument(
        "--find_unused_parameters",
        action="store_true",
        help="enable find_unused_parameters in DDP",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="enable EMA shadow weights during evaluation and checkpointing",
    )
    parser.add_argument(
        "--disable_ema",
        action="store_true",
        help="disable EMA shadow weights",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        help="EMA decay factor",
    )
    parser.add_argument(
        "--ema_update_after_step",
        type=int,
        help="number of optimizer steps before EMA updates start",
    )
    parser.add_argument(
        "--ema_update_every",
        type=int,
        help="EMA update interval in optimizer steps",
    )
    parser.add_argument("--seed", type=int, default=42, help="global random seed")
    parser.add_argument(
        "--seed_list",
        type=int,
        nargs="+",
        help="optional explicit list of seeds for repeated runs",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="load model_save_path and only run evaluation",
    )

    temp_args, _ = parser.parse_known_args()
    config_path = Path(temp_args.config)
    if config_path.exists():
        print(f"Loading the config from '{config_path}' ...")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)

        defaults = {}
        if config:
            for key, value in config.items():
                if isinstance(value, dict):
                    defaults.update(value)
                else:
                    defaults[key] = value
        parser.set_defaults(**defaults)

    args = parser.parse_args()
    if args.disable_online_negative_sampling:
        args.online_negative_sampling = False
    if args.disable_contraindication_negatives:
        args.use_contraindication_negatives = False
    if args.disable_global_router:
        args.use_global_router = False
    if getattr(args, "disable_aux_focal", False):
        args.aux_focal_enabled = False
    if getattr(args, "disable_ema", False):
        args.use_ema = False
    return args


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_enable_nni(args):
    should_enable = bool(args.nni or os.environ.get("NNI_TRIAL_JOB_ID"))
    if not should_enable:
        return args, NullTrialReporter()

    try:
        import nni  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "NNI mode was requested, but the 'nni' package is not installed."
        ) from exc

    optimized_params = nni.get_next_parameter()
    for key, value in optimized_params.items():
        if not hasattr(args, key):
            continue
        setattr(args, key, value)

    if getattr(args, "disable_global_router", False):
        args.use_global_router = False

    trial_id = os.environ.get("NNI_TRIAL_JOB_ID")
    if trial_id:
        model_path = Path(args.model_save_path)
        model_name = model_path.name
        args.model_save_path = str(ROOT / "nni_runs" / trial_id / model_name)

    return args, NNITrialReporter(nni)


def main():
    args = get_args()
    args, trial_reporter = maybe_enable_nni(args)
    dist_state = setup_distributed(args)

    try:
        setup_seed(args.seed + dist_state.rank)

        data_dir = Path(args.data_dir)
        model_save_path = Path(args.model_save_path)
        edge_type_to_predict = ("chemical", "treats", "disease")
        neg_sampler = None

        log0(dist_state, "--- Training Configuration ---")
        if is_main_process(dist_state):
            for key, value in vars(args).items():
                print(f"{key}: {value}")
        log0(dist_state, f"Using device: {dist_state.device}")
        log0(dist_state, "------------------------------")

        log0(dist_state, "Loading data...")
        try:
            train_data = torch.load(
                data_dir / "train_data.pt",
                map_location="cpu",
                weights_only=False,
            )
            valid_data = torch.load(
                data_dir / "val_data.pt",
                map_location="cpu",
                weights_only=False,
            )
            test_data = torch.load(
                data_dir / "test_data.pt",
                map_location="cpu",
                weights_only=False,
            )
        except FileNotFoundError:
            log0(
                dist_state,
                f"Error: data files not found in {data_dir}. Please prepare data first.",
            )
            return

        neg_sampler_path = (
            Path(args.neg_sampler_path)
            if args.neg_sampler_path
            else data_dir / "neg_sampler.pkl"
        )
        if args.online_negative_sampling:
            if neg_sampler_path.exists():
                neg_sampler_mod = importlib.import_module("negative_sampler")
                neg_sampler = neg_sampler_mod.NegativeSampler.load(
                    str(neg_sampler_path)
                )
            else:
                log0(
                    dist_state,
                    f"Warning: neg sampler not found at {neg_sampler_path}; falling back to offline negatives.",
                )
                args.online_negative_sampling = False

        split_mode = str(getattr(train_data, "split_mode", "edge_level")).lower()
        use_train_background_for_eval = split_mode != "disease_cold_start"
        train_data = train_data.to(dist_state.device)

        if use_train_background_for_eval:
            log0(
                dist_state,
                "Aligning val/test message-passing graph with train background edges.",
            )
        else:
            log0(
                dist_state,
                f"Keeping split-specific val/test message-passing graphs (split_mode={split_mode}).",
            )

        seed_list = list(getattr(args, "seed_list", []) or [args.seed])
        if dist_state.distributed and len(seed_list) > 1:
            raise ValueError(
                "Multi-seed repeated runs are not supported with distributed training."
            )

        summaries = []
        for seed_idx, seed in enumerate(seed_list, start=1):
            run_model_save_path = model_save_path
            if len(seed_list) > 1:
                run_model_save_path = model_save_path.with_name(
                    f"{model_save_path.stem}_seed{seed}{model_save_path.suffix}"
                )
            setup_seed(int(seed) + dist_state.rank)

            trainer_cfg = TrainerConfig(
                hidden_channels=args.hidden_channels,
                out_channels=args.out_channels,
                num_srr_layers=args.num_srr_layers,
                num_attention_heads=args.num_attention_heads,
                dropout=args.dropout,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                epochs=args.epochs,
                eval_interval=args.eval_interval,
                log_interval=int(args.log_interval or 1),
                early_stopping_patience=args.early_stopping_patience,
                grad_clip_norm=args.grad_clip_norm,
                monitor_gradients=args.monitor_gradients,
                infonce_temperature=args.infonce_temperature,
                infonce_loss_mode=args.infonce_loss_mode,
                aux_bce_weight=float(args.aux_bce_weight or 0.0),
                aux_focal_enabled=bool(args.aux_focal_enabled),
                aux_focal_alpha=float(args.aux_focal_alpha or 0.25),
                aux_focal_gamma=float(args.aux_focal_gamma or 2.0),
                aux_focal_weight=float(args.aux_focal_weight or 0.0),
                edge_type_to_predict=edge_type_to_predict,
                source_batch_size=int(args.source_batch_size or 256),
                decoder_type=args.decoder_type,
                primary_metric=args.primary_metric,
                ranking_group_axis=str(args.ranking_group_axis or "src"),
                training_group_axis=str(args.training_group_axis or "src"),
                full_ranking_candidate_batch_size=int(
                    args.full_ranking_candidate_batch_size or 65536
                ),
                find_unused_parameters=args.find_unused_parameters,
                use_ema=bool(args.use_ema),
                ema_decay=float(args.ema_decay or 0.999),
                ema_update_after_step=int(args.ema_update_after_step or 0),
                ema_update_every=int(args.ema_update_every or 1),
                edge_dropout_p=float(args.edge_dropout_p or 0.0),
                use_global_router=args.use_global_router,
                mechanism_dropout=args.mechanism_dropout,
                train_neg_per_pos=int(args.train_neg_per_pos or 1),
                online_negative_sampling=bool(args.online_negative_sampling),
                use_contraindication_negatives=bool(
                    args.use_contraindication_negatives
                ),
                negative_curriculum_epochs=int(args.negative_curriculum_epochs or 0),
                negative_start_factor=float(args.negative_start_factor or 1.0),
                hard_negative_ratio_start=float(args.hard_negative_ratio_start or 0.0),
                min_learning_rate_ratio=float(args.min_learning_rate_ratio or 0.1),
            )
            run_reporter = (
                trial_reporter if len(seed_list) == 1 else NullTrialReporter()
            )
            trainer = Trainer(
                trainer_cfg,
                dist_state,
                neg_sampler=neg_sampler,
                trial_reporter=run_reporter,
            )

            if args.eval_only:
                if not run_model_save_path.exists():
                    raise FileNotFoundError(
                        f"Missing checkpoint for eval_only: {run_model_save_path}"
                    )
                log0(
                    dist_state,
                    f"\n--- Start Evaluation Only (seed {seed_idx}/{len(seed_list)} = {seed}) ---",
                )
                model, _, _ = trainer._init_model_and_optimizer(train_data)
                base_model = trainer._base_model(model)
                base_model.load_state_dict(
                    torch.load(
                        str(run_model_save_path),
                        map_location=dist_state.device,
                    )
                )
                test_message_passing_edges = (
                    train_data.edge_index_dict
                    if use_train_background_for_eval
                    else test_data.edge_index_dict
                )
                test_metrics = trainer.evaluate(
                    base_model,
                    test_data,
                    test_message_passing_edges,
                    filtered_positive_pairs=trainer._collect_known_positive_pairs(
                        (train_data, valid_data)
                    ),
                )
                trainer.last_run_summary = {
                    "best_val_score": 0.0,
                    "best_val_metrics": {},
                    "test_metrics": dict(test_metrics),
                }
            else:
                log0(
                    dist_state,
                    f"\n--- Start Training and Evaluation (seed {seed_idx}/{len(seed_list)} = {seed}) ---",
                )
                test_metrics = trainer.train_and_evaluate_model(
                    train_data=train_data,
                    valid_data=valid_data,
                    test_data=test_data,
                    model_save_path=run_model_save_path,
                )
            summary = dict(trainer.last_run_summary or {})
            summaries.append(summary)

            if is_main_process(dist_state):
                log0(dist_state, f"\n--- Final Test Results (seed={seed}) ---")
                for metric, value in test_metrics.items():
                    log0(dist_state, f"  {metric:<12s}: {value:.4f}")

            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        aggregate_summary = aggregate_run_summaries(
            summaries=summaries,
            primary_metric=args.primary_metric,
        )
        if len(seed_list) > 1 and trial_reporter is not None:
            trial_reporter.report_final(aggregate_summary)

        if is_main_process(dist_state) and len(seed_list) > 1:
            log0(dist_state, "\n--- Multi-Seed Summary ---")
            log0(
                dist_state,
                f"  Seeds       : {', '.join(str(seed) for seed in seed_list)}",
            )
            for section_name in ("best_val_metrics", "test_metrics"):
                metrics = aggregate_summary.get(section_name, {})
                pretty_name = (
                    "Best Val" if section_name == "best_val_metrics" else "Test"
                )
                log0(dist_state, f"  {pretty_name}:")
                for metric_name, metric_value in metrics.items():
                    log0(dist_state, f"    {metric_name:<10s}: {metric_value:.4f}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
