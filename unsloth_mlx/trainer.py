"""
Training utilities for Unsloth-MLX

Provides helper functions for loading datasets, training models, and saving results
in standard HuggingFace format.
"""

from typing import Optional, Dict, Any, Union, List
from datasets import load_dataset
import json
from pathlib import Path


def prepare_dataset(
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    split: str = "train",
    formatting_func: Optional[callable] = None,
    **kwargs
):
    """
    Load and prepare dataset for fine-tuning.

    This function provides Unsloth-compatible dataset loading with support for
    HuggingFace datasets library.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "timdettmers/openassistant-guanaco")
        dataset_path: Local path to dataset (JSONL or JSON)
        split: Dataset split to load ("train", "test", "validation")
        formatting_func: Function to format dataset samples
        **kwargs: Additional arguments passed to load_dataset

    Returns:
        Loaded dataset object

    Examples:
        >>> # Load from HuggingFace Hub
        >>> dataset = prepare_dataset("timdettmers/openassistant-guanaco")
        >>>
        >>> # Load from local file
        >>> dataset = prepare_dataset(dataset_path="data/train.jsonl")
        >>>
        >>> # Load with custom split
        >>> dataset = prepare_dataset(
        ...     "yahma/alpaca-cleaned",
        ...     split="train[:1000]"
        ... )
    """

    if dataset_name:
        # Load from HuggingFace Hub
        print(f"Loading dataset '{dataset_name}' from HuggingFace Hub...")
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        print(f"✓ Loaded {len(dataset)} examples")
        return dataset

    elif dataset_path:
        # Load from local file
        dataset_path = Path(dataset_path)
        print(f"Loading dataset from '{dataset_path}'...")

        if dataset_path.suffix == '.jsonl':
            # Load JSONL file
            dataset = load_dataset('json', data_files=str(dataset_path), split='train')
        elif dataset_path.suffix == '.json':
            # Load JSON file
            dataset = load_dataset('json', data_files=str(dataset_path), split='train')
        else:
            raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

        print(f"✓ Loaded {len(dataset)} examples")
        return dataset

    else:
        raise ValueError("Either dataset_name or dataset_path must be provided")


def format_chat_template(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    add_generation_prompt: bool = False,
) -> str:
    """
    Format messages using the model's chat template.

    This function provides Unsloth-compatible chat template formatting, supporting
    different LLM formats (Llama, Mistral, Qwen, etc.).

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        tokenizer: Tokenizer with chat template support
        add_generation_prompt: Whether to add generation prompt at the end

    Returns:
        Formatted prompt string

    Examples:
        >>> messages = [
        ...     {"role": "user", "content": "What is AI?"},
        ...     {"role": "assistant", "content": "AI stands for..."}
        ... ]
        >>> prompt = format_chat_template(messages, tokenizer)
    """

    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False
        )
    else:
        # Fallback to simple formatting if no chat template
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                formatted += f"User: {content}\n"
            elif role == 'assistant':
                formatted += f"Assistant: {content}\n"
            elif role == 'system':
                formatted += f"System: {content}\n"
        if add_generation_prompt:
            formatted += "Assistant: "
        return formatted


def create_training_data(
    dataset: Any,
    tokenizer: Any,
    output_path: str,
    format_type: str = "chat",
    text_field: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> str:
    """
    Create training data file in MLX-LM compatible format.

    Args:
        dataset: Dataset object (from HuggingFace datasets)
        tokenizer: Tokenizer for chat template formatting
        output_path: Path to save formatted data (JSONL format)
        format_type: Data format ("chat", "text", "completions")
        text_field: Field name containing text (for "text" format)
        max_samples: Maximum number of samples to process

    Returns:
        Path to created training data file

    Examples:
        >>> dataset = load_dataset("timdettmers/openassistant-guanaco")
        >>> create_training_data(
        ...     dataset,
        ...     tokenizer,
        ...     "train.jsonl",
        ...     format_type="chat"
        ... )
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples_written = 0
    with open(output_path, 'w') as f:
        for idx, sample in enumerate(dataset):
            if max_samples and idx >= max_samples:
                break

            # Format based on type
            if format_type == "chat":
                # Expect 'messages' field or format it
                if 'messages' in sample:
                    formatted_sample = {"messages": sample['messages']}
                elif 'conversations' in sample:
                    formatted_sample = {"messages": sample['conversations']}
                else:
                    # Try to construct messages from text field
                    continue

            elif format_type == "text":
                # Simple text format
                if text_field and text_field in sample:
                    formatted_sample = {"text": sample[text_field]}
                elif 'text' in sample:
                    formatted_sample = {"text": sample['text']}
                else:
                    continue

            elif format_type == "completions":
                # Prompt-completion format
                if 'prompt' in sample and 'completion' in sample:
                    formatted_sample = {
                        "prompt": sample['prompt'],
                        "completion": sample['completion']
                    }
                else:
                    continue

            else:
                raise ValueError(f"Unsupported format_type: {format_type}")

            f.write(json.dumps(formatted_sample) + '\n')
            samples_written += 1

    print(f"✓ Created training data: {output_path} ({samples_written} samples)")
    return str(output_path)


def save_model_hf_format(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    **kwargs
):
    """
    Save fine-tuned model in standard HuggingFace format.

    This saves the model so that anyone can use it with transformers library,
    not just MLX. Essential for sharing your fine-tuned models!

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        output_dir: Directory to save model
        push_to_hub: Whether to upload to HuggingFace Hub
        repo_id: HuggingFace repo ID (e.g., "username/model-name")
        **kwargs: Additional arguments for pushing to hub

    Examples:
        >>> # Save locally in HF format
        >>> save_model_hf_format(model, tokenizer, "my-finetuned-model")
        >>>
        >>> # Save and push to HuggingFace Hub
        >>> save_model_hf_format(
        ...     model, tokenizer,
        ...     "my-finetuned-model",
        ...     push_to_hub=True,
        ...     repo_id="username/my-model"
        ... )
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving model to {output_dir}...")

    # For MLX models, we need to use mlx_lm utilities to save
    # This will save in a format compatible with HuggingFace
    try:
        from mlx_lm.utils import save_model

        # Save the underlying MLX model
        if hasattr(model, 'model'):
            actual_model = model.model
        else:
            actual_model = model

        save_model(str(output_dir), actual_model, tokenizer)
        print(f"✓ Model saved to {output_dir}")

        if push_to_hub and repo_id:
            print(f"Uploading to HuggingFace Hub: {repo_id}")
            from mlx_lm.utils import upload_to_hub
            upload_to_hub(str(output_dir), repo_id, **kwargs)
            print(f"✓ Model uploaded to {repo_id}")

    except ImportError:
        print("Warning: mlx_lm.utils not available. Attempting alternative save method...")
        # Alternative: save tokenizer at minimum
        tokenizer.save_pretrained(str(output_dir))
        print(f"✓ Tokenizer saved to {output_dir}")


def export_to_gguf(
    model_path: str,
    output_path: Optional[str] = None,
    quantization: str = "q4_k_m",
    adapter_path: Optional[str] = None,
    **kwargs
):
    """
    Export model to GGUF format for use with llama.cpp, Ollama, etc.

    This function uses mlx_lm.fuse to merge adapters (if any) and export to GGUF.

    Args:
        model_path: Path to the base model or HuggingFace model ID
        output_path: Path for output GGUF file (defaults to model_path/model.gguf)
        quantization: Quantization type (q4_k_m, q5_k_m, q8_0, f16, etc.)
        adapter_path: Path to LoRA adapters to fuse before export
        **kwargs: Additional export options

    Examples:
        >>> # Export base model to GGUF
        >>> export_to_gguf("mlx-community/Llama-3.2-1B-Instruct-4bit")
        >>>
        >>> # Export fine-tuned model with adapters
        >>> export_to_gguf(
        ...     "mlx-community/Llama-3.2-1B-Instruct-4bit",
        ...     adapter_path="./adapters",
        ...     output_path="my-model.gguf",
        ... )
    """
    import subprocess

    model_path = Path(model_path) if not model_path.startswith(('mlx-', 'meta-', 'mistral')) else model_path
    if output_path is None:
        if isinstance(model_path, Path):
            output_path = model_path / "model.gguf"
        else:
            output_path = Path("./model.gguf")
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting model to GGUF format...")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_path}")
    if adapter_path:
        print(f"  Adapters: {adapter_path}")

    # Build mlx_lm.fuse command
    cmd = [
        "mlx_lm.fuse",
        "--model", str(model_path),
        "--export-gguf",
        "--gguf-path", str(output_path),
    ]

    # Add adapter path if provided
    if adapter_path:
        cmd.extend(["--adapter-path", str(adapter_path)])

    # Add de-quantize flag for better quality (optional)
    if kwargs.get('de_quantize', False):
        cmd.append("--de-quantize")

    print(f"\nRunning: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Model exported to {output_path}")
        return str(output_path)

    except subprocess.CalledProcessError as e:
        print(f"Error during GGUF export: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")

        # Try alternative method using convert
        print("\nTrying alternative export method...")
        try:
            alt_cmd = [
                "mlx_lm.convert",
                "--hf-path", str(model_path),
                "-q",  # Quantize
                "--export-gguf",
            ]
            subprocess.run(alt_cmd, check=True)
            print(f"✓ Model exported using alternative method")
            return str(output_path)
        except Exception as alt_e:
            print(f"Alternative method also failed: {alt_e}")
            print("\nManual export command:")
            print(f"  mlx_lm.fuse --model {model_path} --export-gguf --gguf-path {output_path}")
            raise


def get_training_config(
    output_dir: str = "./lora_finetuned",
    num_train_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    **kwargs
) -> Dict[str, Any]:
    """
    Get recommended training configuration.

    Returns a configuration dict compatible with MLX-LM training.

    Args:
        output_dir: Directory to save trained model
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size for training
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        **kwargs: Additional training arguments

    Returns:
        Training configuration dict

    Examples:
        >>> config = get_training_config(
        ...     num_train_epochs=5,
        ...     learning_rate=1e-4
        ... )
    """

    config = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": kwargs.get("lora_dropout", 0.05),
        "warmup_steps": kwargs.get("warmup_steps", 100),
        "max_seq_length": kwargs.get("max_seq_length", 2048),
        "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 1),
        "save_steps": kwargs.get("save_steps", 500),
        "logging_steps": kwargs.get("logging_steps", 10),
    }

    config.update(kwargs)
    return config
