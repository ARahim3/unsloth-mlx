"""
Unsloth-MLX: MLX-powered LLM fine-tuning for Apple Silicon

A drop-in replacement for Unsloth that uses Apple's MLX framework instead of CUDA/Triton kernels.

Supported Training Methods:
- SFT (Supervised Fine-Tuning)
- DPO (Direct Preference Optimization)
- ORPO (Odds Ratio Preference Optimization)
- GRPO (Group Relative Policy Optimization) - DeepSeek R1 style
- KTO (Kahneman-Tversky Optimization)
- SimPO (Simple Preference Optimization)
- VLM (Vision Language Model) fine-tuning
"""

__version__ = "0.3.0"  # Native training + proper DPO/ORPO/GRPO/KTO/SimPO losses

from unsloth_mlx.model import FastLanguageModel
from unsloth_mlx.trainer import (
    prepare_dataset,
    format_chat_template,
    create_training_data,
    save_model_hf_format,
    export_to_gguf,
    get_training_config,
)
from unsloth_mlx.sft_trainer import SFTTrainer, SFTConfig, TrainingArguments

# RL Trainers
from unsloth_mlx.rl_trainers import (
    DPOTrainer,
    DPOConfig,
    ORPOTrainer,
    ORPOConfig,
    GRPOTrainer,
    GRPOConfig,
    KTOTrainer,
    SimPOTrainer,
    prepare_preference_dataset,
    create_reward_function,
)

# Loss functions for custom training
from unsloth_mlx.losses import (
    compute_log_probs,
    compute_log_probs_with_lengths,
    dpo_loss,
    orpo_loss,
    kto_loss,
    simpo_loss,
    sft_loss,
    grpo_loss,
    grpo_batch_loss,
    compute_reference_logprobs,
)

# Vision Language Models
from unsloth_mlx.vlm import (
    FastVisionModel,
    VLMSFTTrainer,
    load_vlm_dataset,
)

# Chat Templates and Dataset Formatting (Unsloth-compatible)
from unsloth_mlx.chat_templates import (
    detect_dataset_format,
    standardize_sharegpt,
    convert_to_mlx_format,
    get_formatting_func,
    apply_chat_template_to_sample,
    alpaca_to_text,
)

__all__ = [
    # Core
    "FastLanguageModel",
    "__version__",
    # SFT Training
    "SFTTrainer",
    "SFTConfig",
    "TrainingArguments",
    # RL Trainers
    "DPOTrainer",
    "DPOConfig",
    "ORPOTrainer",
    "ORPOConfig",
    "GRPOTrainer",
    "GRPOConfig",
    "KTOTrainer",
    "SimPOTrainer",
    # Vision Models
    "FastVisionModel",
    "VLMSFTTrainer",
    # Loss Functions
    "compute_log_probs",
    "compute_log_probs_with_lengths",
    "dpo_loss",
    "orpo_loss",
    "kto_loss",
    "simpo_loss",
    "sft_loss",
    "grpo_loss",
    "grpo_batch_loss",
    "compute_reference_logprobs",
    # Utilities
    "prepare_dataset",
    "prepare_preference_dataset",
    "format_chat_template",
    "create_training_data",
    "save_model_hf_format",
    "export_to_gguf",
    "get_training_config",
    "create_reward_function",
    "load_vlm_dataset",
    # Chat Templates and Dataset Formatting
    "detect_dataset_format",
    "standardize_sharegpt",
    "convert_to_mlx_format",
    "get_formatting_func",
    "apply_chat_template_to_sample",
    "alpaca_to_text",
]
