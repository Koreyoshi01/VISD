# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import contextmanager
from configs.data_root import DATA_ROOT

ROOT = os.path.join(DATA_ROOT, "videos")
GQA_ROOT = os.path.join(ROOT, "gqa")
TIMERFT_ROOT = os.path.join(ROOT, "timerft")
TVG_ROOT = os.path.join(ROOT, "tvg_r1")
VIDEO_ESPRESSO_KF_ROOT = os.path.join(ROOT, "videoespresso/kfs")
VIDEO_ESPRESSO_ROOT = os.path.join(ROOT, "videoespresso/videos")
STR_KF_ROOT = os.path.join(ROOT, "stgr/temporal_grounding/kfs")
STR_DATA = os.path.join(ROOT, "stgr/temporal_grounding/videos")
STR_PLM_KF_ROOT = os.path.join(ROOT, "stgr/plm/kfs")
STR_PLM_DATA = os.path.join(ROOT, "stgr/plm/videos")
GENERAL_VIDEO_ROOT = os.path.join(ROOT, "videor1")

import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import random
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.utils.data
import transformers
from transformers.utils import import_utils as hf_import_utils
from packaging import version

# This training path can load local JSON/JSONL data without HuggingFace datasets.
# On the current DSW node, importing `transformers.Trainer` while datasets is
# installed is the most common EMFILE trigger because Trainer eagerly imports
# datasets/pandas/pyarrow. Keep this opt-in so normal environments are unchanged.
if str(os.getenv("PHASE2_FORCE_SKIP_HF_DATASETS_IMPORT", "")).strip().lower() in {"1", "true", "yes", "on"}:
    hf_import_utils._datasets_available = False

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save_file

from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from src.open_r1.vision_process import process_vision_info
from src.open_r1.phase2.phase2_rollout_advantage import compute_rollout_advantages
from src.open_r1.phase2.phase2_rollout_logging import get_reward_func_name

import copy
from PIL import Image
import numpy as np

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    from transformers.integrations import WandbCallback

try:
    from transformers import AutoVideoProcessor, Qwen2VLImageProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
except ImportError:
    AutoVideoProcessor = None
    Qwen2VLImageProcessor = None
    Qwen2_5_VLForConditionalGeneration = None
    Qwen2_5_VLProcessor = None
    

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def is_conversational(example: dict[str, Any]) -> bool:
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages"]
    example_keys = {key for key in example.keys() if key in supported_keys}
    if not example_keys:
        return False
    key = example_keys.pop()
    maybe_messages = example[key]
    if not isinstance(maybe_messages, list) or not maybe_messages:
        return False
    maybe_message = maybe_messages[0]
    return isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message


def maybe_apply_chat_template(
    example: dict[str, Any],
    tokenizer,
) -> dict[str, Any]:
    if not is_conversational(example):
        return example

    chat_template_owner = tokenizer
    if getattr(chat_template_owner, "chat_template", None) is None:
        nested_tokenizer = getattr(chat_template_owner, "tokenizer", None)
        if nested_tokenizer is not None and getattr(nested_tokenizer, "chat_template", None) is not None:
            chat_template_owner = nested_tokenizer

    output = dict(example)
    if "prompt" in example:
        prompt_messages = example["prompt"]
        last_role = prompt_messages[-1]["role"]
        if last_role == "user":
            prompt = chat_template_owner.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        elif last_role == "assistant":
            prompt = chat_template_owner.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        else:
            raise ValueError(f"Invalid role in the last message: {last_role}")
        output["prompt"] = prompt
    if "messages" in example:
        output["text"] = chat_template_owner.apply_chat_template(example["messages"], tokenize=False)
    return output


def load_multimodal_processing_class(model_id: str, *, max_pixels: Optional[int], min_pixels: Optional[int]):
    try:
        processing_class = AutoProcessor.from_pretrained(model_id)
    except ValueError as exc:
        if "Unrecognized image processor" not in str(exc):
            raise
        if AutoVideoProcessor is None or Qwen2VLImageProcessor is None or Qwen2_5_VLProcessor is None:
            raise ImportError("Qwen2.5-VL processing classes are unavailable in this transformers build") from exc

        image_processor = Qwen2VLImageProcessor.from_pretrained(model_id)
        video_processor = AutoVideoProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processing_class = Qwen2_5_VLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
        )

    pad_token_id = processing_class.tokenizer.pad_token_id
    processing_class.pad_token_id = pad_token_id
    processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
    if hasattr(processing_class, "image_processor"):
        processing_class.image_processor.max_pixels = max_pixels
        processing_class.image_processor.min_pixels = min_pixels
    return processing_class


CANONICAL_METRIC_MAP = {
    "loss": "opt/loss",
    "grad_norm": "opt/grad_norm",
    "learning_rate": "opt/lr",
    "kl": "opt/kl",
    "reward": "reward/total",
    "reward_std": "reward/std",
    "completion_length": "sample/completion_length",
    "all_wrong": "sample/all_wrong_fraction",
    "all_correct": "sample/all_correct_fraction",
    "rewards/ans_acc_reward": "reward/base/ans_acc",
    "rewards/ans_tiou_reward": "reward/base/ans_tiou",
    "rewards/ans_viou_reward": "reward/base/ans_viou",
    "rewards/thk_temporal_point_reward": "reward/base/temporal_point",
    "rewards/thk_temporal_segment_reward": "reward/base/temporal_segment",
    "rewards/thk_spatial_reward": "reward/base/spatial",
    "rewards/format_reward": "reward/base/format",
    "phase2/base_reward_disabled": "reward/base/disabled",
    "phase2/process_feedback_enabled": "judge/enabled",
    "phase2/process_feedback_requested_fraction": "judge/requested_fraction",
    "phase2/process_feedback_available_fraction": "judge/generated_feedback_available_fraction",
    "phase2/process_feedback_error_fraction": "judge/error_fraction",
    "phase2/process_feedback_reused_from_reward_fraction": "judge/reused_fraction",
    "phase2/judge_feedback_available_fraction": "judge/feedback_available_fraction",
    "phase2/requested_sample_fraction": "routing/requested_fraction",
    "phase2/effective_sample_fraction": "routing/effective_fraction",
    "phase2/target_sample_fraction": "routing/target_fraction",
    "phase2/reweight_sample_fraction": "routing/to_reweight_fraction",
    "phase2/group_all_success_fraction": "routing/group_all_success_fraction",
    "phase2/group_all_failure_fraction": "routing/group_all_failure_fraction",
    "phase2/group_mixed_fraction": "routing/group_mixed_fraction",
    "phase2/group_avg_reward": "routing/group_avg_reward",
    "phase2/group_avg_target_score": "routing/group_avg_target_score",
    "phase2/requested_target_sample_fraction": "routing/requested_target_fraction",
    "phase2/effective_target_sample_fraction": "routing/effective_target_fraction",
    "phase2/effective_reweight_sample_fraction": "routing/effective_reweight_fraction",
    "phase2/target_score_mean": "routing/target_score_mean",
    "phase2/teacher_enabled": "teacher/enabled",
    "phase2/teacher_hindsight_available": "teacher/hindsight_available",
    "phase2/teacher_targeted": "teacher/targeted_fraction",
    "phase2/teacher_triggered": "teacher/triggered_fraction",
    "phase2/teacher_forward_sample_fraction": "teacher/forward_fraction",
    "phase2/teacher_replay_dropped_count": "teacher/replay_dropped_count",
    "phase2/teacher_prompt_length": "teacher/prompt_length",
    "phase2/vision_read_failed_local_fraction": "sample/vision_read_failed_local_fraction",
    "phase2/vision_read_failed_loss_skipped_fraction": "sample/vision_read_failed_loss_skipped_fraction",
    "phase2/vision_media_missing_local_fraction": "sample/vision_media_missing_local_fraction",
    "phase2/vision_decode_failed_local_fraction": "sample/vision_decode_failed_local_fraction",
    "phase2/vision_unknown_failed_local_fraction": "sample/vision_unknown_failed_local_fraction",
    "phase2/vision_read_failed_count": "sample/vision_read_failed_count",
    "phase2/vision_media_missing_count": "sample/vision_media_missing_count",
    "phase2/vision_decode_failed_count": "sample/vision_decode_failed_count",
    "phase2/vision_unknown_failed_count": "sample/vision_unknown_failed_count",
    "phase2/teacher_gap_mean": "teacher/gap_mean",
    "phase2/teacher_gap_std": "teacher/gap_std",
    "phase2/teacher_gap_abs_mean": "teacher/gap_abs_mean",
    "phase2/teacher_gap_positive_fraction": "teacher/gap_positive_fraction",
    "phase2/teacher_gap_front_mean": "teacher/gap_front_mean",
    "phase2/teacher_gap_mid_mean": "teacher/gap_mid_mean",
    "phase2/teacher_gap_tail_mean": "teacher/gap_tail_mean",
    "phase2/teacher_gap_tail_minus_front": "teacher/gap_tail_minus_front",
    "phase2/teacher_entropy_mean": "teacher/entropy_mean",
    "phase2/teacher_entropy_gap_mean": "teacher/entropy_gap_mean",
    "phase2/teacher_entropy_gap_abs_mean": "teacher/entropy_gap_abs_mean",
    "phase2/effective_token_fraction": "teacher/effective_token_fraction",
    "phase2/effective_token_fraction_within_target": "teacher/effective_token_fraction_within_target",
    "phase2/hindsight_available_fraction": "sample/hindsight_available_fraction",
    "phase2/feedback_available_fraction": "sample/feedback_available_fraction",
    "phase2/semantic_score_available_fraction": "sample/semantic_score_available_fraction",
    "phase2/response_entropy_mean": "sample/response_entropy_mean",
    "phase2/response_entropy_std": "sample/response_entropy_std",
    "phase2/response_entropy_front_mean": "sample/response_entropy_front_mean",
    "phase2/response_entropy_mid_mean": "sample/response_entropy_mid_mean",
    "phase2/response_entropy_tail_mean": "sample/response_entropy_tail_mean",
    "phase2/response_entropy_tail_minus_front": "sample/response_entropy_tail_minus_front",
    "phase2/answer_tag_present_fraction": "sample/answer_tag_present_fraction",
    "phase2/answer_tag_missing_fraction": "sample/answer_tag_missing_fraction",
    "phase2/incomplete_answer_tag_fraction": "sample/incomplete_answer_tag_fraction",
    "phase2/truncated_fraction": "sample/truncated_fraction",
    "phase2/empty_output_fraction": "sample/empty_output_fraction",
    "phase2/format_success_fraction": "sample/format_success_fraction",
    "phase2/format_failure_fraction": "sample/format_failure_fraction",
}

WANDB_TOP_LEVEL_GROUP_PREFIXES = (
    "opt/",
    "reward/",
    "rollout/",
    "judge/",
    "routing/",
    "teacher/",
    "sample/",
    "focus/",
)

FOCUS_METRIC_MAP = {
    "reward/base/ans_acc": "focus/answer_semantic",
    "reward/base/ans_tiou": "focus/answer_window",
    "reward/base/ans_viou": "focus/answer_box",
    "reward/base/temporal_point": "focus/reasoning_temporal_point",
    "reward/base/temporal_segment": "focus/reasoning_temporal_segment",
    "reward/base/spatial": "focus/reasoning_spatial",
    "reward/base/format": "focus/format",
    "rollout/group_all_zero_fraction": "focus/rollout_all_zero_fraction",
    "rollout/group_any_nonzero_fraction": "focus/rollout_any_nonzero_fraction",
    "teacher/gap_abs_mean": "focus/teacher_gap_abs_mean",
    "teacher/gap_positive_fraction": "focus/teacher_gap_positive_fraction",
    "teacher/entropy_gap_abs_mean": "focus/teacher_entropy_gap_abs_mean",
    "sample/response_entropy_mean": "focus/response_entropy_mean",
    "sample/response_entropy_tail_minus_front": "focus/response_entropy_tail_minus_front",
    "opt/kl": "focus/kl",
    "sample/completion_length": "focus/completion_length",
    "sample/truncated_fraction": "focus/truncated_fraction",
    "sample/format_success_fraction": "focus/format_success_fraction",
    "sample/no_parsed_answer_fraction": "focus/no_parsed_answer_fraction",
}

CANONICAL_ALIAS_MAP = {
    "sample/response_entropy_mean": "sample/student_entropy_mean",
    "sample/response_entropy_front_mean": "sample/student_entropy_front_mean",
    "sample/response_entropy_mid_mean": "sample/student_entropy_mid_mean",
    "sample/response_entropy_tail_mean": "sample/student_entropy_tail_mean",
    "sample/response_entropy_tail_minus_front": "sample/student_entropy_tail_minus_front",
    "sample/answer_tag_missing_fraction": "sample/no_parsed_answer_fraction",
    "sample/format_success_fraction": "sample/format_parse_success_fraction",
}


def canonicalize_user_facing_metrics(logs: dict[str, float]) -> dict[str, float]:
    canonical: dict[str, float] = {}

    def set_metric(key: str, value):
        canonical[key] = value

    for key, value in logs.items():
        mapped_key = CANONICAL_METRIC_MAP.get(key)
        if mapped_key is not None:
            set_metric(mapped_key, value)
            continue

        if key in {"epoch", "train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss"}:
            set_metric(key, value)
            continue

        if key.startswith("rollout/"):
            set_metric(key, value)
            continue

        if key.startswith("phase2/teacher_signal_mode_"):
            suffix = key.removeprefix("phase2/teacher_signal_mode_")
            set_metric(f"routing/teacher_signal_mode/{suffix}", value)
            continue

        if key.startswith("phase2/reweighting_"):
            suffix = key.removeprefix("phase2/reweighting_")
            set_metric(f"routing/reweighting/{suffix}", value)
            continue

        if key.startswith("reward/") or key.startswith("judge/") or key.startswith("routing/") or key.startswith("teacher/") or key.startswith("sample/") or key.startswith("opt/"):
            set_metric(key, value)
            continue

        set_metric(key, value)

    for source_key, alias_key in CANONICAL_ALIAS_MAP.items():
        if source_key in canonical and alias_key not in canonical:
            canonical[alias_key] = canonical[source_key]

    for source_key, focus_key in FOCUS_METRIC_MAP.items():
        if source_key in canonical and focus_key not in canonical:
            canonical[focus_key] = canonical[source_key]

    return canonical


def expand_user_facing_metric_aliases(logs: dict[str, float]) -> dict[str, float]:
    return canonicalize_user_facing_metrics(logs)


def rewrite_phase2_wandb_logs(logs: dict[str, Any]) -> dict[str, Any]:
    rewritten: dict[str, Any] = {}
    for key, value in logs.items():
        if key.startswith("eval_"):
            rewritten[f"eval/{key[len('eval_'):]}"] = value
            continue
        if key.startswith("test_"):
            rewritten[f"test/{key[len('test_'):]}"] = value
            continue
        if key.startswith(("train/", "eval/", "test/")):
            rewritten[key] = value
            continue
        if key == "rollout/latest_samples" or key.startswith(WANDB_TOP_LEVEL_GROUP_PREFIXES):
            rewritten[key] = value
            continue
        rewritten[f"train/{key}"] = value
    return rewritten


if is_wandb_available():
    class Phase2GroupedWandbCallback(WandbCallback):
        def on_log(self, args, state, control, model=None, logs=None, **kwargs):
            single_value_scalars = [
                "train_runtime",
                "train_samples_per_second",
                "train_steps_per_second",
                "train_loss",
                "total_flos",
            ]

            if self._wandb is None:
                return
            if not self._initialized:
                self.setup(args, state, model)
            if state.is_world_process_zero:
                for key, value in logs.items():
                    if key in single_value_scalars:
                        self._wandb.run.summary[key] = value
                non_scalar_logs = {key: value for key, value in logs.items() if key not in single_value_scalars}
                non_scalar_logs = rewrite_phase2_wandb_logs(non_scalar_logs)
                self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step})
else:
    Phase2GroupedWandbCallback = None


def install_grouped_wandb_callback(trainer: Trainer) -> None:
    if not is_wandb_available() or Phase2GroupedWandbCallback is None:
        return
    if not hasattr(trainer, "callback_handler") or trainer.callback_handler is None:
        return

    default_wandb_callbacks = [
        callback
        for callback in trainer.callback_handler.callbacks
        if isinstance(callback, WandbCallback) and not isinstance(callback, Phase2GroupedWandbCallback)
    ]
    if not default_wandb_callbacks:
        return

    for callback in default_wandb_callbacks:
        trainer.remove_callback(callback)
    if not any(isinstance(callback, Phase2GroupedWandbCallback) for callback in trainer.callback_handler.callbacks):
        trainer.add_callback(Phase2GroupedWandbCallback)


@contextmanager
def temporarily_set_model_eval(model):
    was_training = bool(getattr(model, "training", False))
    if was_training:
        model.eval()
    try:
        yield model
    finally:
        if was_training:
            model.train()


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[torch.utils.data.Dataset, torch.utils.data.IterableDataset]] = None,
        eval_dataset: Optional[
            Union[
                torch.utils.data.Dataset,
                torch.utils.data.IterableDataset,
                dict[str, Union[torch.utils.data.Dataset, torch.utils.data.IterableDataset]],
            ]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        gspo = True,
        phase2_enable_ref_model: bool = True,
        phase2_enable_teacher_model: bool = False,
        phase2_teacher_update_mode: str = "off",
        phase2_teacher_update_rate: float = 0.05,
        phase2_teacher_update_interval: int = 10,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
            

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                from transformers import Qwen2VLForConditionalGeneration

                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                from transformers import AriaForConditionalGeneration

                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
                # model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        self.phase2_enable_ref_model = bool(phase2_enable_ref_model)
        self.phase2_enable_teacher_model = bool(phase2_enable_teacher_model)
        self.phase2_teacher_update_mode = str(phase2_teacher_update_mode or "off").lower()
        self.phase2_teacher_update_rate = float(phase2_teacher_update_rate)
        self.phase2_teacher_update_interval = max(1, int(phase2_teacher_update_interval))

        if self.phase2_teacher_update_mode not in {"off", "copy", "ema"}:
            raise ValueError(
                "phase2_teacher_update_mode must be one of {'off', 'copy', 'ema'}, "
                f"got {self.phase2_teacher_update_mode!r}"
            )
        if not 0.0 <= self.phase2_teacher_update_rate <= 1.0:
            raise ValueError(
                "phase2_teacher_update_rate must be within [0, 1], "
                f"got {self.phase2_teacher_update_rate}"
            )

        self.ref_model = self._build_phase2_reference_model(
            model=model,
            model_id=model_id,
            model_init_kwargs=model_init_kwargs,
            peft_config=peft_config,
        )
        self.teacher_model = self._build_phase2_teacher_model(
            model=model,
            model_id=model_id,
            model_init_kwargs=model_init_kwargs,
            peft_config=peft_config,
        )

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or True:
                processing_class = load_multimodal_processing_class(
                    model_id,
                    max_pixels=max_pixels,
                    min_pixels=min_pixels,
                )
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        pad_token_id = getattr(processing_class, "pad_token_id", None)
        if pad_token_id is None and hasattr(processing_class, "tokenizer"):
            pad_token_id = processing_class.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(processing_class, "eos_token_id", None)
        if pad_token_id is None and hasattr(processing_class, "tokenizer"):
            pad_token_id = processing_class.tokenizer.eos_token_id
        if pad_token_id is not None and hasattr(processing_class, "pad_token_id"):
            processing_class.pad_token_id = pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes
        self.script_args = script_args

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        # self.temporal = script_args.temporal
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )

        # self.len_control = script_args.len_control
        self.beta = args.beta

        self.epsilon_low = 0.2
        self.epsilon_high = 0.2
        self.gspo = gspo
        configured_reward_weights = getattr(args, "reward_weights", None)
        if configured_reward_weights is not None and len(configured_reward_weights) not in {0, len(self.reward_funcs)}:
            raise ValueError(
                "Number of reward weights must match number of reward functions, "
                f"got {len(configured_reward_weights)} vs {len(self.reward_funcs)}."
            )
        self.reward_weights = configured_reward_weights
        self.reward_names = [get_reward_func_name(reward_func) for reward_func in self.reward_funcs]

        self.total_data = len(train_dataset)* args.num_train_epochs

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        install_grouped_wandb_callback(self)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if self.teacher_model is not None:
            if self.is_deepspeed_enabled:
                self.teacher_model = prepare_deepspeed(self.teacher_model, self.accelerator)
            else:
                self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)
            self._copy_student_to_teacher()

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _load_policy_model_from_pretrained(self, model_id: str, model_init_kwargs: dict[str, Any]):
        if "Qwen2-VL" in model_id:
            from transformers import Qwen2VLForConditionalGeneration

            return Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        if "Qwen2.5-VL" in model_id:
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        if "Aria" in model_id:
            from transformers import AriaForConditionalGeneration

            return AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)

    @staticmethod
    def _freeze_model_weights(model: Optional[torch.nn.Module]) -> Optional[torch.nn.Module]:
        if model is None:
            return None
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "parameters"):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        return model

    def _build_phase2_reference_model(
        self,
        *,
        model,
        model_id: str,
        model_init_kwargs: dict[str, Any],
        peft_config,
    ):
        if not self.phase2_enable_ref_model:
            return None
        if is_deepspeed_zero3_enabled():
            return self._freeze_model_weights(self._load_policy_model_from_pretrained(model_id, model_init_kwargs))
        if peft_config is None:
            return self._freeze_model_weights(create_reference_model(model))
        return None

    def _build_phase2_teacher_model(
        self,
        *,
        model,
        model_id: str,
        model_init_kwargs: dict[str, Any],
        peft_config,
    ):
        if not self.phase2_enable_teacher_model:
            return None
        if peft_config is not None:
            return self._freeze_model_weights(copy.deepcopy(model))
        if is_deepspeed_zero3_enabled():
            return self._freeze_model_weights(self._load_policy_model_from_pretrained(model_id, model_init_kwargs))
        return self._freeze_model_weights(create_reference_model(model))

    def _get_student_update_model(self):
        return self.model_wrapped if getattr(self, "model_wrapped", None) is not None else self.model

    def _get_unwrapped_teacher_model(self):
        if self.teacher_model is None:
            return None
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and hasattr(accelerator, "unwrap_model"):
            try:
                return accelerator.unwrap_model(self.teacher_model, keep_torch_compile=False)
            except TypeError:
                return accelerator.unwrap_model(self.teacher_model)
            except Exception:
                pass
        return getattr(self.teacher_model, "module", self.teacher_model)

    @staticmethod
    def _teacher_checkpoint_dir(output_dir: str) -> str:
        return os.path.join(output_dir, "teacher_model")

    @staticmethod
    def _teacher_checkpoint_file(output_dir: str) -> str:
        return os.path.join(Qwen2VLGRPOTrainer._teacher_checkpoint_dir(output_dir), "teacher_model.safetensors")

    @staticmethod
    def _copy_tensor_into_target(target_tensor: torch.Tensor, source_tensor: torch.Tensor) -> None:
        source_data = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
        if isinstance(target_tensor, torch.nn.Parameter) and Qwen2VLGRPOTrainer._is_zero3_partitioned_param(target_tensor):
            import deepspeed

            modifier_rank = 0
            with deepspeed.zero.GatheredParameters([target_tensor], modifier_rank=modifier_rank):
                if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == modifier_rank:
                    target_tensor.data.copy_(source_data)
            return
        target_tensor.data.copy_(source_data)

    @staticmethod
    def _dedupe_state_dict_for_safetensors(state_dict: dict[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
        deduped = OrderedDict()
        seen = set()
        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            storage_ptr = tensor.untyped_storage().data_ptr() if tensor.numel() > 0 else None
            key = (storage_ptr, tensor.storage_offset(), tuple(tensor.shape), str(tensor.dtype))
            if key in seen:
                continue
            seen.add(key)
            deduped[name] = tensor.detach().cpu().contiguous()
        return deduped

    def _save_teacher_model_checkpoint(self, output_dir: str) -> None:
        if self.teacher_model is None:
            return

        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and hasattr(accelerator, "get_state_dict"):
            state_dict = accelerator.get_state_dict(self.teacher_model)
        else:
            state_dict = self.teacher_model.state_dict()

        if not getattr(self.args, "should_save", True):
            return

        teacher_dir = self._teacher_checkpoint_dir(output_dir)
        os.makedirs(teacher_dir, exist_ok=True)
        teacher_file = self._teacher_checkpoint_file(output_dir)
        deduped_state_dict = self._dedupe_state_dict_for_safetensors(state_dict)
        safetensors_save_file(deduped_state_dict, teacher_file, metadata={"format": "pt"})

        teacher_model = self._get_unwrapped_teacher_model()
        teacher_config = getattr(teacher_model, "config", None)
        if teacher_config is not None and hasattr(teacher_config, "save_pretrained"):
            teacher_config.save_pretrained(teacher_dir)
        generation_config = getattr(teacher_model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "save_pretrained"):
            generation_config.save_pretrained(teacher_dir)

    def _load_teacher_model_checkpoint(self, checkpoint_dir: str) -> bool:
        if self.teacher_model is None:
            return False

        teacher_file = self._teacher_checkpoint_file(checkpoint_dir)
        if not os.path.exists(teacher_file):
            return False

        teacher_model = self._get_unwrapped_teacher_model()
        if teacher_model is None:
            return False

        named_parameters = dict(teacher_model.named_parameters())
        named_buffers = dict(teacher_model.named_buffers())

        with safe_open(teacher_file, framework="pt", device="cpu") as checkpoint_reader:
            for name in checkpoint_reader.keys():
                tensor = checkpoint_reader.get_tensor(name)
                target = named_parameters.get(name)
                if target is not None:
                    self._copy_tensor_into_target(target, tensor)
                    continue
                buffer = named_buffers.get(name)
                if buffer is not None:
                    self._copy_tensor_into_target(buffer, tensor)

        return True

    @staticmethod
    def _is_zero3_partitioned_param(param: torch.nn.Parameter) -> bool:
        return hasattr(param, "ds_id") or hasattr(param, "ds_status")

    @staticmethod
    def _update_param_data(
        target_param: torch.nn.Parameter,
        source_param: torch.nn.Parameter,
        *,
        mode: str,
        update_rate: float,
    ) -> None:
        def apply_update():
            source_data = source_param.data.to(device=target_param.device, dtype=target_param.dtype)
            if mode == "copy":
                target_param.data.copy_(source_data)
            elif mode == "ema":
                target_param.data.mul_(1.0 - update_rate).add_(source_data, alpha=update_rate)
            else:
                raise ValueError(f"Unsupported teacher update mode: {mode}")

        if Qwen2VLGRPOTrainer._is_zero3_partitioned_param(target_param) or Qwen2VLGRPOTrainer._is_zero3_partitioned_param(source_param):
            import deepspeed

            gathered_params = [target_param, source_param]
            modifier_rank = 0
            with deepspeed.zero.GatheredParameters(gathered_params, modifier_rank=modifier_rank):
                if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == modifier_rank:
                    apply_update()
            return

        apply_update()

    @classmethod
    def _copy_module_state(cls, target_model, source_model) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
                cls._update_param_data(
                    target_param,
                    source_param,
                    mode="copy",
                    update_rate=1.0,
                )
            for target_buffer, source_buffer in zip(target_model.buffers(), source_model.buffers()):
                target_buffer.data.copy_(source_buffer.data.to(device=target_buffer.device, dtype=target_buffer.dtype))

    @classmethod
    def _ema_update_module_state(cls, target_model, source_model, update_rate: float) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
                cls._update_param_data(
                    target_param,
                    source_param,
                    mode="ema",
                    update_rate=update_rate,
                )
            for target_buffer, source_buffer in zip(target_model.buffers(), source_model.buffers()):
                if torch.is_floating_point(target_buffer):
                    source_data = source_buffer.data.to(device=target_buffer.device, dtype=target_buffer.dtype)
                    target_buffer.data.mul_(1.0 - update_rate).add_(source_data, alpha=update_rate)
                else:
                    target_buffer.data.copy_(
                        source_buffer.data.to(device=target_buffer.device, dtype=target_buffer.dtype)
                    )

    def _copy_student_to_teacher(self) -> bool:
        if self.teacher_model is None:
            return False
        self._copy_module_state(self.teacher_model, self._get_student_update_model())
        return True

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        resolved_output_dir = output_dir if output_dir is not None else self.args.output_dir
        self._save_teacher_model_checkpoint(resolved_output_dir)

    def _maybe_update_teacher_model(self, *, completed_step: int) -> bool:
        if not self.phase2_enable_teacher_model or self.teacher_model is None:
            return False
        if not bool(getattr(self, "phase2_teacher_model_update_enabled", True)):
            return False
        if self.phase2_teacher_update_mode == "off":
            return False
        if completed_step <= 0 or completed_step % self.phase2_teacher_update_interval != 0:
            return False

        if self.phase2_teacher_update_mode == "copy":
            self._copy_module_state(self.teacher_model, self._get_student_update_model())
        elif self.phase2_teacher_update_mode == "ema":
            self._ema_update_module_state(
                self.teacher_model,
                self._get_student_update_model(),
                update_rate=self.phase2_teacher_update_rate,
            )
        else:
            raise ValueError(f"Unsupported phase2_teacher_update_mode: {self.phase2_teacher_update_mode}")

        self._metrics["phase2/teacher_model_update_applied"].append(1.0)
        self._metrics["phase2/teacher_model_update_step"].append(float(completed_step))
        self._metrics["phase2/teacher_model_update_mode_copy"].append(
            1.0 if self.phase2_teacher_update_mode == "copy" else 0.0
        )
        self._metrics["phase2/teacher_model_update_mode_ema"].append(
            1.0 if self.phase2_teacher_update_mode == "ema" else 0.0
        )
        return True

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get token log probabilities and optional entropy for the trailing tokens we train on.
    def _get_per_token_logps_and_entropy(
        self,
        model,
        input_ids,
        *,
        logits_to_keep=None,
        compute_entropy=True,
        return_token_logits=False,
        **kwargs,
    ):
        logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        if logits_to_keep is not None:
            logits_to_keep = int(logits_to_keep)
            if logits_to_keep <= 0:
                raise ValueError(f"logits_to_keep must be positive, got {logits_to_keep}")
            logits = logits[:, -logits_to_keep:, :]
            input_ids = input_ids[:, -logits_to_keep:]

        log_probs = logits.log_softmax(dim=-1)
        per_token_logps = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        per_token_entropy = None
        if compute_entropy:
            per_token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        if return_token_logits:
            return per_token_logps, per_token_entropy, logits
        return per_token_logps, per_token_entropy

    def _get_per_token_logps(self, model, input_ids, *, logits_to_keep=None, **kwargs):
        per_token_logps, _ = self._get_per_token_logps_and_entropy(
            model,
            input_ids,
            logits_to_keep=logits_to_keep,
            compute_entropy=False,
            **kwargs,
        )
        return per_token_logps
    
    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def _move_prompt_inputs_to_device(
        self, prompt_inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in prompt_inputs.items()
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        input_copy = [copy.deepcopy(inputs[0]['prompt'][1])]

        if inputs[0]['source'] == 'videoespresso_train_video':
            video_root = VIDEO_ESPRESSO_ROOT
            input_copy[0]['content'][0]['video'] =  os.path.join(video_root, inputs[0]['video_path'])
        elif inputs[0]['source'] == 'timerft':
            video_root = TIMERFT_ROOT
            input_copy[0]['content'][0]['video'] =  os.path.join(video_root, inputs[0]['video_path'])
        elif inputs[0]['source'] == 'gqa':
            image_root = GQA_ROOT
            input_copy[0]['content'][0]['image'] =  os.path.join(image_root, inputs[0]['image_path'])
        elif "STR" in inputs[0]['source']:
            if "STR_plm" in inputs[0]['source']:
                video_root = STR_PLM_DATA
            else:
                video_root = STR_DATA
            input_copy[0]['content'][0]['video'] =  os.path.join(video_root, inputs[0]['video_path'])
        elif "TVG" in inputs[0]['source']:
            video_root = TVG_ROOT
            input_copy[0]['content'][0]['video'] =  os.path.join(video_root, inputs[0]['video_path'])
        elif "videor1" in inputs[0]['source']:
            video_root = GENERAL_VIDEO_ROOT
            input_copy[0]['content'][0]['video'] =  os.path.join(video_root, inputs[0]['video_path'])

        else:
            raise ValueError(f"Invalid source: {inputs[0]['source']}")
        
        input_copy = self.remove_none_from_data(input_copy)
    
        # remove None from key_items
        if 'key_items' in inputs[0]:
            keys_to_remove = []
            for key, item in inputs[0]['key_items'].items():
                if item is None:
                    keys_to_remove.append(key)
                elif isinstance(item, dict):
                    sub_keys_to_remove = [k for k, v in item.items() if v is None]
                    for k in sub_keys_to_remove:
                        del item[k]            
            for key in keys_to_remove:
                del inputs[0]['key_items'][key]
            
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
            if image_inputs is not None:
                inputs[0]['image_size_refine'] = (image_inputs[0].size[0], image_inputs[0].size[1])  # W * H
                inputs[0]['prompt_text_final'] = prompts_text[0]

            if video_inputs is not None:
                inputs[0]['video_sample_fps'] = video_kwargs['fps'][0]
                inputs[0]['video_duration'] = video_inputs[0].size(0) / video_kwargs['fps'][0]
                inputs[0]['image_size'] = (video_inputs[0].size(3), video_inputs[0].size(2))  # W * H
                inputs[0]['prompt_text_final'] = prompts_text[0]

        except Exception as e:
            print(f"process_vision_info error, using fixed data, {e}")
            
        current_step = self.state.global_step + 1
        total_steps = self.state.max_steps
        inputs[0]['step_percent'] = current_step/total_steps
        # print("STEP:", current_step, total_steps, inputs[0]['step_percent'])

        multi_image = True

        if video_inputs is None:
            multi_image = False
            
        if multi_image:
            if inputs[0]['task'] != "temporal-spatial free-form QA":
                frame_prompt = ""
                ori_idx = 0
                while ori_idx < len(video_inputs[0]):
                    time_now = round(ori_idx / video_kwargs['fps'][0],1)
                    frame_prompt += f"Frame {ori_idx + 1} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"    
                    ori_idx += 1
                frame_prompt += f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"

                prompts_text[0] = prompts_text[0].replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)
                inputs[0]['prompt_text_final'] = prompts_text[0]
                image_inputs = [video_inputs[0]]
            
            else:
                width, height = video_inputs[0].size(3), video_inputs[0].size(2)
                image_size = (width, height)

                # Here, we need to add key frames.
                if inputs[0]['source'] == 'videoespresso_train_video':
                    key_frame_root = VIDEO_ESPRESSO_KF_ROOT
                elif 'STR_plm' in inputs[0]['source']:
                    key_frame_root = STR_PLM_KF_ROOT
                else:
                    key_frame_root = STR_KF_ROOT
                
                key_frames = []

                for key_frame in inputs[0]["key_frames"]:
                    kf_path = os.path.join(key_frame_root, key_frame["path"])
                    kf = Image.open(kf_path)
                    kf = kf.convert('RGB')
                    resized_kf = kf.resize(image_size)
                    resized_kf= np.array(resized_kf)
                    resized_kf = np.transpose(resized_kf, (2, 0, 1))
                    resized_kf = torch.from_numpy(resized_kf)
                    key_frames.append((round(key_frame["time"]), resized_kf))
                
                frame_prompt = ""
                refined_image_inputs = []
                kf_idx = 0
                ori_idx = 0
                frame_idx = 1
                while ori_idx < len(video_inputs[0]):
                    time_now = int(ori_idx / video_kwargs['fps'][0])
                    if kf_idx < len(key_frames) and time_now >= key_frames[kf_idx][0]:
                        refined_image_inputs.append(key_frames[kf_idx][1])
                        time_now = round(key_frames[kf_idx][0],1)
                        frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"     
                        kf_idx += 1
                    else:
                        refined_image_inputs.append(video_inputs[0][ori_idx])
                        time_now = round(ori_idx / video_kwargs['fps'][0],1)
                        frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n" 
                        ori_idx += 1
                    frame_idx += 1
                frame_prompt += f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"
                image_inputs = torch.stack(refined_image_inputs)
                # print(image_inputs.shape)
                image_inputs = [image_inputs]
                prompts_text[0] = prompts_text[0].replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)
                inputs[0]['prompt_text_final'] = prompts_text[0]

            prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=None,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )

        else:
            # frame_prompt = f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"
            # prompts_text[0] = prompts_text[0].replace("<|vision_start|><|video_pad|><|vision_end|>", "<|vision_start|><|video_pad|><|vision_end|>" + frame_prompt)
            # print(prompts_text[0])
            
            prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                **video_kwargs,
            )
        
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs = self._move_prompt_inputs_to_device(prompt_inputs)

        # fix prompt_inputs["input_ids"] length issue
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]     
        
        # Generate completions
        with temporarily_set_model_eval(model):
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
                prompt_length = prompt_ids.size(1)
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
                prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            
        # print('prompt_length:', prompt_length)    

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")

        if multi_image or image_inputs is not None:
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)
        else:
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)

        if 'second_per_grid_ts' in prompt_inputs:
            del prompt_inputs["second_per_grid_ts"]
 
        try:
            per_token_logps = self._get_per_token_logps(
                model,
                prompt_completion_ids,
                logits_to_keep=completion_ids.size(1),
                **prompt_inputs,
            )
        except Exception as e:
            print(f"Error computing per_token_logps: {e}. Setting output to zero.")
            per_token_logps = self._get_per_token_logps(
                model,
                prompt_completion_ids,
                logits_to_keep=completion_ids.size(1),
            )
        
        with torch.inference_mode():
            if self.phase2_enable_ref_model:
                try:
                    if self.ref_model is not None:
                        ref_per_token_logps = self._get_per_token_logps(
                            self.ref_model,
                            prompt_completion_ids,
                            logits_to_keep=completion_ids.size(1),
                            **prompt_inputs,
                        )
                    else:
                        with self.accelerator.unwrap_model(model).disable_adapter():
                            ref_per_token_logps = self._get_per_token_logps(
                                model,
                                prompt_completion_ids,
                                logits_to_keep=completion_ids.size(1),
                                **prompt_inputs,
                            )
                except Exception as e:
                    print(f"Error computing ref_per_token_logps: {e}. Setting output to zero.")
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            model,
                            prompt_completion_ids,
                            logits_to_keep=completion_ids.size(1),
                        )
                x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
                per_token_kl = torch.exp(x_clamped) - x_clamped - 1
            else:
                per_token_kl = torch.zeros_like(per_token_logps)
        
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
            
        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    # Repeat each value in the column for `num_generations` times
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        rewards = rewards_per_func.sum(dim=1) # 等比例相加
        
        # if self.len_control:
        #     mem_rewards = [0] * self.num_generations
        #     mask = rewards_per_func[:, 0] > 0.1
        #     lenth_list = completion_mask.sum(1)
        #     selected_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
                    
        #     if len(selected_indices) > 1:     
        #         for idx in selected_indices:
        #             if 320 <= lenth_list[idx] <= 512:
        #                 rewards[idx] += 0.2
        
        # print(rewards)
        # print(completion_mask.sum(1))

        # Compute grouped-wise GRPO advantages.
        advantages, rewards, mean_grouped_rewards, std_grouped_rewards = self._compute_rollout_advantages(
            rewards_per_func,
        )
        
        # x - x.detach() allows for preserving gradients from x
        ## the original code of video-r1
        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # refined by mjh start

        log_ratio = per_token_logps - per_token_logps.detach()
        if self.gspo:
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            log_importance_weights = log_ratio
        
        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

        # refined by mjh end

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        
        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        
        num_devices = gathered_rewards.size(0) // self.num_generations 
        rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices
        
        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices
        
        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # print("num_devices:", num_devices, "gathered_rewards.size(0):", gathered_rewards.size(0))

        return loss

    def _compute_rollout_advantages(self, rewards_per_func: torch.Tensor):
        return compute_rollout_advantages(
            rewards_per_func=rewards_per_func,
            num_generations=self.num_generations,
            reward_weights=getattr(self, "reward_weights", None),
        )

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        logs = canonicalize_user_facing_metrics(logs)
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        from trl.trainer.utils import generate_model_card, get_comet_experiment_url

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
