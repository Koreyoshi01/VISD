import os
os.environ.setdefault("WANDB_MODE", "offline")
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
from configs.data_root import DATA_ROOT

ROOT = os.path.join(DATA_ROOT, "videos")
GQA_ROOT = os.path.join(ROOT, "gqa")
TIMERFT_ROOT = os.path.join(ROOT, "timerft")
TVG_ROOT = os.path.join(ROOT, "tvg_r1")
VIDEO_ESPRESSO_ROOT = os.path.join(ROOT, "videoespresso/videos")
VIDEO_ESPRESSO_KF_ROOT = os.path.join(ROOT, "videoespresso/kfs")
STR_DATA = os.path.join(ROOT, "stgr/temporal_grounding/videos")
STR_PLM_DATA = os.path.join(ROOT, "stgr/plm/videos")
GENERAL_VIDEO_ROOT = os.path.join(ROOT, "videor1")

from dataclasses import dataclass, field
from typing import Optional
from trl.scripts.utils import ScriptArguments, TrlParser
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.model_config import ModelConfig
from src.open_r1.phase2.phase2_runtime_callbacks import (
    Phase2CheckpointScheduleConfig,
    Phase2RuntimeScheduleCallback,
    Phase2RuntimeScheduleConfig,
)
from src.open_r1.runtime_compat import enable_torch_checkpoint_resume_compat
from src.open_r1.phase2.trainer_routing import (
    normalize_phase2_advantage_clip,
    resolve_phase2_trainer_mode,
)
# from src.open_r1.reward_func import accuracy_reward, format_reward, temporal_reward, spatial_reward
from src.open_r1.reward_func import ans_acc_reward, ans_tiou_reward, ans_viou_reward, thk_temporal_point_reward, thk_temporal_segment_reward, thk_spatial_reward, format_reward


def ensure_model_init_kwargs_precision(training_args):
    model_init_kwargs = dict(getattr(training_args, "model_init_kwargs", None) or {})

    if "torch_dtype" not in model_init_kwargs:
        if getattr(training_args, "bf16", False):
            model_init_kwargs["torch_dtype"] = "bfloat16"
        elif getattr(training_args, "fp16", False):
            model_init_kwargs["torch_dtype"] = "float16"

    training_args.model_init_kwargs = model_init_kwargs
    return model_init_kwargs


def get_peft_config(model_args):
    if getattr(model_args, "use_peft", False) is False:
        return None

    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ValueError(
            "You need to have PEFT library installed in your environment, make sure to install `peft`."
        ) from exc

    return LoraConfig(
        task_type=model_args.lora_task_type,
        r=model_args.lora_r,
        target_modules=model_args.lora_target_modules,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_rslora=model_args.use_rslora,
        use_dora=model_args.use_dora,
        modules_to_save=model_args.lora_modules_to_save,
    )


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["ans_acc", "ans_tiou", "ans_viou", "thk_temporal_point", "thk_temporal_segment", "thk_spatial", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    json_loader: Optional[str] = field(
        default="hf",
        metadata={"help": "JSON dataset loader for local .json/.jsonl files: 'hf' (STGR default) or 'simple'."},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    spatial_iou_mode: Optional[str] = field(
        default="max",
        metadata={"help": "IoU aggregation mode for spatial reward across multiple GT objects: 'max' (default) or 'avg'"},
    )
    identity_match_mode: Optional[str] = field(
        default="none",
        metadata={"help": "Object identity matching mode: 'none' (ignore identity), 'soft' (flexible matching), 'strict' (exact match)"},
    )
    spatial_norm_mode: Optional[str] = field(
        default="all",
        metadata={"help": "Spatial reward normalization: 'all' (divide by all claims), 'matched' (divide by matched claims only)"},
    )
    correct_tempgate: Optional[bool] = field(
        default=True,
        metadata={"help": "Temporal gating condition: True (abs diff <= threshold), False (gt - pred < threshold)"},
    )
    # Ablation: exclude grounding rewards
    wo_spatial: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, exclude thk_spatial reward (spatial grounding ablation)"},
    )
    wo_tempspatial: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, exclude thk_temporal_point + thk_temporal_segment + thk_spatial (full grounding ablation)"},
    )
    wo_acc: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, exclude ans_acc reward (answer accuracy ablation)"},
    )
    enable_phase2_teacher: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable the VISD feedback-conditioned teacher-replay trainer."},
    )
    phase2_enable_ref_model: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to instantiate and use the reference model for KL computation."},
    )
    phase2_enable_teacher_model: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to keep a standalone teacher model resident for phase2 teacher replay."},
    )
    phase2_teacher_update_mode: Optional[str] = field(
        default="off",
        metadata={"help": "Standalone teacher update mode: 'off', 'copy', or 'ema'."},
    )
    phase2_teacher_update_rate: Optional[float] = field(
        default=0.05,
        metadata={"help": "EMA update rate for the standalone teacher model."},
    )
    phase2_teacher_update_interval: Optional[int] = field(
        default=10,
        metadata={"help": "Update the standalone teacher model every N optimization steps."},
    )
    phase2_teacher_trainer_module: Optional[str] = field(
        default="",
        metadata={
            "help": "Python module path for the standalone phase2 teacher trainer class "
                    "(e.g. open_r1.phase2.grpo_trainer_phase2_teacher). "
                    "Default: use open_r1.phase2.grpo_trainer_phase2_teacher. "
                    "Class name must be Qwen2VLGRPOTrainerPhase2Teacher."
        },
    )
    phase2_teacher_feedback_text: Optional[str] = field(
        default="",
        metadata={"help": "Optional extra corrective feedback text injected into the phase2 hindsight teacher prompt."},
    )
    phase2_teacher_success_reward_threshold: Optional[float] = field(
        default=0.999,
        metadata={"help": "Success threshold used for reporting group routing state."},
    )
    phase2_teacher_success_metric: Optional[str] = field(
        default="task_aware",
        metadata={"help": "Success metric used for phase2 targeting: 'task_aware', 'answer_only', or 'native_total'."},
    )
    phase2_teacher_include_text_feedback: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether phase2 teacher prompts should include textual feedback already present in the sample."},
    )
    phase2_teacher_include_reward_breakdown_in_feedback: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether phase2 teacher prompts should append reward breakdown summaries as feedback text."},
    )
    phase2_base_reward_mode: Optional[str] = field(
        default="native",
        metadata={"help": "Base reward mode for VISD RL: 'native' keeps task rewards, 'none' disables them."},
    )
    phase2_inject_teacher_signal: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to inject teacher signal into the phase2 RL backbone."},
    )
    phase2_importance_sampling_level: Optional[str] = field(
        default="sequence",
        metadata={
            "help": "Policy importance sampling level for the phase2 RL backbone: "
            "'token'/'grpo', 'sequence'/'gspo', or 'sequence_token'/'gspo_token'."
        },
    )
    phase2_reweighting_mixing_lambda: Optional[float] = field(
        default=1.0,
        metadata={"help": "reweighting mixing lambda."},
    )
    phase2_reweighting_weight_mode: Optional[str] = field(
        default="sampled",
        metadata={"help": "reweighting weight mode: 'sampled' or 'topk_interpolate'."},
    )
    phase2_reweighting_topk: Optional[int] = field(
        default=8,
        metadata={"help": "Teacher top-k size for reweighting top-k interpolation mode."},
    )
    phase2_reweighting_topk_gamma: Optional[float] = field(
        default=1.0,
        metadata={"help": "Gamma for reweighting top-k interpolation: 1.0=sampled, 0.0=top-k."},
    )
    phase2_reweighting_weight_clip: Optional[float] = field(
        default=0.2,
        metadata={"help": "reweighting weight clip. Set negative to disable clipping."},
    )
    phase2_reweighting_anneal_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Linearly anneal reweighting mixing lambda to 0 across this many steps. <=0 disables annealing."},
    )
    phase2_teacher_disable_after_step: Optional[int] = field(
        default=0,
        metadata={"help": "Disable teacher signal injection after this step. <=0 keeps it enabled."},
    )
    phase2_process_feedback_disable_after_step: Optional[int] = field(
        default=0,
        metadata={"help": "Disable judge/process feedback requests after this step. <=0 keeps them enabled."},
    )
    phase2_runtime_step_offset: Optional[int] = field(
        default=0,
        metadata={"help": "Logical training-step offset used by phase2 runtime scheduling for dry-run/resume validation."},
    )
    phase2_checkpoint_dense_until_step: Optional[int] = field(
        default=0,
        metadata={"help": "Use dense checkpoint interval up to and including this step. <=0 disables dense scheduling."},
    )
    phase2_checkpoint_dense_interval: Optional[int] = field(
        default=0,
        metadata={"help": "Checkpoint interval during the dense phase. <=0 disables dense periodic saving."},
    )
    phase2_checkpoint_sparse_interval: Optional[int] = field(
        default=0,
        metadata={"help": "Checkpoint interval after the dense phase. <=0 disables sparse periodic saving."},
    )
    phase2_teacher_max_prompt_length: Optional[int] = field(
        default=18432,
        metadata={"help": "Maximum prompt length for teacher replay prompts."},
    )
    phase2_process_feedback_enable: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable judge process-feedback generation from current sampled outputs."},
    )
    phase2_process_feedback_timeout: Optional[float] = field(
        default=30.0,
        metadata={"help": "Timeout in seconds for judge process-feedback requests."},
    )
    phase2_process_feedback_scope: Optional[str] = field(
        default="target_only",
        metadata={"help": "Process-feedback request scope: 'target_only' or 'all_with_hindsight'."},
    )
    phase2_process_feedback_model: Optional[str] = field(
        default=None,
        metadata={"help": "Optional model override for process-feedback judge requests."},
    )
    phase2_process_feedback_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "Optional API base URL override for process-feedback judge requests."},
    )
    phase2_process_feedback_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Optional API key override for process-feedback judge requests."},
    )
    phase2_process_feedback_max_feedback_chars: Optional[int] = field(
        default=1000,
        metadata={"help": "Maximum feedback characters returned by the judge process-feedback request."},
    )
    phase2_teacher_gamma: Optional[float] = field(
        default=1.0,
        metadata={"help": "Discount factor used when converting teacher bonuses into token advantages."},
    )
    phase2_reward_metadata_enable: Optional[bool] = field(
        default=True,
        metadata={"help": "Ask reward functions to emit dict-style metadata when supported."},
    )
    phase2_rollout_trace_enable: Optional[bool] = field(
        default=False,
        metadata={"help": "Write rollout-level traces and curated observability logs for phase2 training."},
    )
    phase2_rollout_trace_wandb_steps: Optional[int] = field(
        default=10,
        metadata={"help": "How often to log rollout trace samples to W&B tables when tracing is enabled."},
    )
    phase2_rollout_trace_max_groups: Optional[int] = field(
        default=1,
        metadata={"help": "Maximum rollout groups to persist per logging step when tracing is enabled."},
    )
reward_funcs_registry = {
    "ans_acc": ans_acc_reward,
    "ans_tiou": ans_tiou_reward,
    "ans_viou": ans_viou_reward,
    "thk_temporal_point": thk_temporal_point_reward,
    "thk_temporal_segment": thk_temporal_segment_reward,
    "thk_spatial": thk_spatial_reward,
    "format": format_reward
}


def _prewarm_optional_runtime_dependencies() -> None:
    # On the current DSW node, loading decord only after the model has already
    # opened many shared libraries is much more likely to hit EMFILE/libxcb
    # failures. Preloading it early is a lightweight runtime workaround.
    try:
        import decord  # noqa: F401
    except Exception as exc:
        print(f"[Phase2] Optional runtime prewarm skipped for decord: {exc}")


def _resolve_phase2_arg(script_args, key: str, default):
    value = getattr(script_args, key, None)
    return value if value is not None else default


def _safe_config_repr(config):
    sensitive_markers = ("api_key", "token", "password", "secret")
    values = dict(vars(config))
    for key in list(values):
        if any(marker in key.lower() for marker in sensitive_markers):
            values[key] = "***REDACTED***"
    return f"{config.__class__.__name__}({values})"





def main(script_args, training_args, model_args):
    # Delay the heavy trainer/data imports until we actually execute training.
    from src.open_r1.phase2.data import get_data
    from src.open_r1.phase2.grpo_trainer import Qwen2VLGRPOTrainer

    _prewarm_optional_runtime_dependencies()

    # Get reward functions (wo_spatial/wo_tempspatial ablation applied in trainer reward computation)
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = get_data(script_args)
    enable_phase2_teacher = bool(script_args.enable_phase2_teacher)

    trainer_mode = resolve_phase2_trainer_mode(
        enable_phase2_teacher=enable_phase2_teacher,
    )

    # ============================================================
    # Phase2 trainer routing
    # ============================================================
    if trainer_mode == "phase2_teacher":
        from src.open_r1.phase2.grpo_trainer_phase2_teacher import Qwen2VLGRPOTrainerPhase2Teacher

        trainer_cls = Qwen2VLGRPOTrainerPhase2Teacher
        print("[Phase2] VISD teacher-replay trainer enabled.")
        print(
            f"[Phase2Teacher] Config: success_threshold={script_args.phase2_teacher_success_reward_threshold}, "
            f"importance_sampling_level={script_args.phase2_importance_sampling_level}, "
            f"reweighting_weight_mode={script_args.phase2_reweighting_weight_mode}, "
            f"reweighting_topk={script_args.phase2_reweighting_topk}"
        )
    else:
        trainer_cls = Qwen2VLGRPOTrainer  # if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified

    print("using: ", trainer_cls)
    print("training_args:", _safe_config_repr(training_args))
    print("script_args:", _safe_config_repr(script_args))
    print("model_args:", _safe_config_repr(model_args))

    ensure_model_init_kwargs_precision(training_args)

    # Initialize the GRPO trainer
    # Build common kwargs
    trainer_kwargs = dict(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        phase2_enable_ref_model=_resolve_phase2_arg(
            script_args,
            "phase2_enable_ref_model",
            True,
        ),
        phase2_enable_teacher_model=_resolve_phase2_arg(
            script_args,
            "phase2_enable_teacher_model",
            False,
        ),
        phase2_teacher_update_mode=_resolve_phase2_arg(
            script_args,
            "phase2_teacher_update_mode",
            "off",
        ),
        phase2_teacher_update_rate=_resolve_phase2_arg(
            script_args,
            "phase2_teacher_update_rate",
            0.05,
        ),
        phase2_teacher_update_interval=_resolve_phase2_arg(
            script_args,
            "phase2_teacher_update_interval",
            10,
        ),
    )

    if trainer_mode == "phase2_teacher":
        trainer_kwargs.update(
            phase2_teacher_success_metric=script_args.phase2_teacher_success_metric,
            phase2_teacher_feedback_text=script_args.phase2_teacher_feedback_text or None,
            phase2_base_reward_mode=script_args.phase2_base_reward_mode,
            phase2_inject_teacher_signal=script_args.phase2_inject_teacher_signal,
            phase2_importance_sampling_level=script_args.phase2_importance_sampling_level,
            phase2_reweighting_mixing_lambda=script_args.phase2_reweighting_mixing_lambda,
            phase2_reweighting_weight_mode=script_args.phase2_reweighting_weight_mode,
            phase2_reweighting_topk=script_args.phase2_reweighting_topk,
            phase2_reweighting_topk_gamma=script_args.phase2_reweighting_topk_gamma,
            phase2_reweighting_weight_clip=normalize_phase2_advantage_clip(
                script_args.phase2_reweighting_weight_clip
            ),
            phase2_teacher_max_prompt_length=script_args.phase2_teacher_max_prompt_length,
            phase2_process_feedback_enable=script_args.phase2_process_feedback_enable,
            phase2_process_feedback_timeout=script_args.phase2_process_feedback_timeout,
            phase2_process_feedback_scope=script_args.phase2_process_feedback_scope,
            phase2_process_feedback_model=script_args.phase2_process_feedback_model,
            phase2_process_feedback_base_url=script_args.phase2_process_feedback_base_url,
            phase2_process_feedback_api_key=script_args.phase2_process_feedback_api_key,
            phase2_process_feedback_max_feedback_chars=script_args.phase2_process_feedback_max_feedback_chars,
            phase2_teacher_gamma=script_args.phase2_teacher_gamma,
            phase2_reward_metadata_enable=script_args.phase2_reward_metadata_enable,
            phase2_rollout_trace_enable=script_args.phase2_rollout_trace_enable,
            phase2_rollout_trace_wandb_steps=script_args.phase2_rollout_trace_wandb_steps,
            phase2_rollout_trace_max_groups=script_args.phase2_rollout_trace_max_groups,
        )

    trainer = trainer_cls(**trainer_kwargs)
    runtime_callback = Phase2RuntimeScheduleCallback(
        trainer=trainer,
        runtime_config=Phase2RuntimeScheduleConfig(
            reweighting_initial_lambda=float(getattr(trainer, "phase2_reweighting_mixing_lambda", 1.0)),
            reweighting_anneal_steps=max(0, int(_resolve_phase2_arg(
                script_args,
                "phase2_reweighting_anneal_steps",
                0,
            ))),
            teacher_disable_after_step=max(0, int(_resolve_phase2_arg(
                script_args,
                "phase2_teacher_disable_after_step",
                0,
            ))),
            process_feedback_initial_enabled=bool(getattr(trainer, "phase2_process_feedback_enable", True)),
            process_feedback_disable_after_step=max(0, int(_resolve_phase2_arg(
                script_args,
                "phase2_process_feedback_disable_after_step",
                0,
            ))),
            runtime_step_offset=int(_resolve_phase2_arg(
                script_args,
                "phase2_runtime_step_offset",
                0,
            )),
        ),
        checkpoint_config=Phase2CheckpointScheduleConfig(
            dense_until_step=max(0, int(_resolve_phase2_arg(
                script_args,
                "phase2_checkpoint_dense_until_step",
                0,
            ))),
            dense_interval=max(0, int(_resolve_phase2_arg(
                script_args,
                "phase2_checkpoint_dense_interval",
                0,
            ))),
            sparse_interval=max(0, int(_resolve_phase2_arg(
                script_args,
                "phase2_checkpoint_sparse_interval",
                0,
            ))),
        ),
    )
    trainer.add_callback(runtime_callback)
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        enable_torch_checkpoint_resume_compat()
        maybe_load_teacher_checkpoint = getattr(trainer, "_load_teacher_model_checkpoint", None)
        if callable(maybe_load_teacher_checkpoint):
            teacher_restored = bool(maybe_load_teacher_checkpoint(checkpoint))
            print(f"[Phase2Teacher] Restore resident teacher checkpoint: {teacher_restored} from {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
