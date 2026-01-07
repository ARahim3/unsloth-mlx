# Quick Start Guide - Unsloth-MLX

Get started with Unsloth-MLX in under 5 minutes!

## Installation

### Prerequisites
- Apple Silicon Mac (M1, M2, M3, M4, or M5)
- macOS 13.0+ (macOS 15.0+ recommended for large models)
- Python 3.9+
- 16GB+ RAM (32GB+ recommended for larger models)

### Install Unsloth-MLX

```bash
pip install unsloth-mlx
```

Or install from source:

```bash
git clone https://github.com/yourusername/unsloth-mlx.git
cd unsloth-mlx
pip install -e .
```

## Basic Usage

### 1. Load a Model

```python
from unsloth_mlx import FastLanguageModel

# Load a quantized model from MLX community
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

print("Model loaded successfully!")
```

### 2. Configure LoRA for Fine-Tuning

```python
# Add LoRA adapters for parameter-efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

print("LoRA adapters configured!")
```

### 3. Run Inference

```python
# Enable inference mode for optimal performance
FastLanguageModel.for_inference(model)

# Prepare your prompt
prompt = "What is the capital of France?"
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)

# Generate response
from mlx_lm import generate

response = generate(
    model.model,
    tokenizer,
    prompt=formatted_prompt,
    max_tokens=100,
)

print(f"Response: {response}")
```

## Migrating from Unsloth

If you have existing Unsloth code, migration is simple - just change the import!

**Before (Unsloth with CUDA):**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

**After (Unsloth-MLX with MLX):**
```python
from unsloth_mlx import FastLanguageModel  # ← Only change!

# For MLX, use mlx-community models
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Llama-3.2-8B-Instruct-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

## Fine-Tuning Workflow

### Step 1: Prepare Your Dataset

Create a JSONL file with your training data:

```jsonl
{"messages": [{"role": "user", "content": "What is ML?"}, {"role": "assistant", "content": "Machine Learning is..."}]}
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "Artificial Intelligence is..."}]}
```

### Step 2: Load and Configure Model

```python
from unsloth_mlx import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)
```

### Step 3: Fine-Tune with MLX-LM

Use MLX-LM's command-line tool for training:

```bash
mlx_lm.lora \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --train \
    --data train.jsonl \
    --iters 1000 \
    --learning-rate 1e-5 \
    --lora-layers 16
```

### Step 4: Use the Fine-Tuned Model

```python
# Load model with adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Load your fine-tuned adapters
# (MLX-LM saves adapters in adapters/ directory by default)

FastLanguageModel.for_inference(model)

# Generate with your fine-tuned model!
```

## Examples

Check out the `examples/` directory for more:

- `01_simple_loading.py` - Basic model loading
- `02_lora_configuration.py` - LoRA setup
- `03_inference.py` - Text generation
- `04_simple_finetuning.py` - Complete fine-tuning workflow

## Tips & Best Practices

### Model Selection

For quantized models, always prefer `mlx-community` models from HuggingFace:
- These are pre-quantized and optimized for MLX
- Available in 4-bit quantization for memory efficiency
- Examples: `mlx-community/Llama-3.2-3B-Instruct-4bit`

### Memory Management

**For 16GB RAM:**
- Use 1B-3B parameter models with 4-bit quantization
- Example: `mlx-community/Llama-3.2-1B-Instruct-4bit`

**For 32GB RAM:**
- Use up to 7B parameter models with 4-bit quantization
- Example: `mlx-community/Llama-3.2-7B-Instruct-4bit`

**For 48GB+ RAM:**
- Use 7B-13B parameter models with 4-bit quantization
- Consider 8-bit for larger models

### LoRA Configuration

**For experimentation:**
```python
r=8, lora_alpha=16  # Fewer parameters, faster
```

**For quality:**
```python
r=16, lora_alpha=32  # More parameters, better results
```

**For production:**
```python
r=32, lora_alpha=64  # Best quality
```

### Inference Optimization

Always call `FastLanguageModel.for_inference(model)` before generation:
- Enables KV caching for faster generation
- Disables dropout layers
- Applies MLX-specific optimizations

## Troubleshooting

### Model Not Found
**Error:** `Model 'xyz' not found`

**Solution:** Use MLX community models:
```python
# ❌ Don't use CUDA-specific models
model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

# ✅ Use MLX community models
model_name = "mlx-community/Llama-3.2-8B-Instruct-4bit"
```

### Out of Memory
**Error:** System runs out of memory

**Solutions:**
1. Use a smaller model (1B or 3B parameters)
2. Use 4-bit quantization
3. Reduce `max_seq_length`
4. Close other applications

### Slow Generation
**Issue:** Text generation is slow

**Solutions:**
1. Call `FastLanguageModel.for_inference(model)` before generation
2. Use 4-bit quantized models
3. Reduce `max_tokens` in generation
4. Ensure macOS 15+ for memory wiring optimization

## Next Steps

- Read the [API Reference](api_reference.md) for detailed documentation
- Check [Migration Guide](migration_guide.md) for moving from Unsloth
- Browse [examples/](../examples/) for more use cases
- Join the community discussions

## Getting Help

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review [examples/](../examples/) for working code
3. Open an issue on GitHub
4. Read MLX documentation for MLX-specific questions

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM GitHub](https://github.com/ml-explore/mlx-lm)
- [Unsloth Documentation](https://docs.unsloth.ai)
- [HuggingFace MLX Community](https://huggingface.co/mlx-community)

---

**Happy fine-tuning! 🚀**
