"""
Example 5: Complete Fine-Tuning Workflow with HuggingFace Dataset

This example demonstrates the FULL workflow:
1. Load a model from ANY HuggingFace repo
2. Load a dataset from HuggingFace
3. Prepare data for training
4. Configure LoRA
5. Train the model (using MLX-LM)
6. Save in HuggingFace format
7. Export to GGUF

This shows how Unsloth-MLX works just like Unsloth!
"""

from unsloth_mlx import (
    FastLanguageModel,
    prepare_dataset,
    format_chat_template,
    create_training_data,
    save_model_hf_format,
    export_to_gguf,
    get_training_config,
)


def main():
    print("=" * 70)
    print("Unsloth-MLX Example: Complete Fine-Tuning Workflow")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load Model from ANY HuggingFace Repository
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Load Model from HuggingFace")
    print("=" * 70)

    print("\nYou can load ANY model from HuggingFace:")
    print("  - meta-llama/Llama-3.2-1B-Instruct")
    print("  - mistralai/Mistral-7B-Instruct-v0.3")
    print("  - Qwen/Qwen2.5-7B-Instruct")
    print("  - Or pre-quantized: mlx-community/Llama-3.2-1B-Instruct-4bit")
    print("\nFor this example, we'll use a pre-quantized model (faster):")

    model, tokenizer = FastLanguageModel.from_pretrained(
        # Can be ANY HuggingFace model!
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    print("✓ Model loaded successfully!")
    print(f"  Model: {model.model_name}")
    print(f"  Max sequence length: {model.max_seq_length}")

    # =========================================================================
    # Step 2: Load Dataset from HuggingFace
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Load Dataset from HuggingFace")
    print("=" * 70)

    print("\nLoading a real dataset from HuggingFace Hub...")
    print("You can use:")
    print("  - prepare_dataset('yahma/alpaca-cleaned')")
    print("  - prepare_dataset('mlabonne/FineTome-100k')")
    print("  - prepare_dataset(dataset_path='local/data.jsonl')")
    print("  - Any dataset compatible with HuggingFace datasets!")

    # For demo, we'll create a small sample dataset
    # In real use: dataset = prepare_dataset("timdettmers/openassistant-guanaco")

    sample_data = [
        {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity and readability."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a branch of AI that enables systems to learn from data."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is deep learning?"},
                {"role": "assistant", "content": "Deep learning uses neural networks with multiple layers to learn complex patterns."}
            ]
        },
    ]

    print("✓ Dataset loaded (3 sample examples)")
    print("  In production, you'd load thousands of examples")

    # =========================================================================
    # Step 3: Prepare Training Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Prepare Training Data")
    print("=" * 70)

    # Create training data file
    import json
    train_file = "train_data.jsonl"
    with open(train_file, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')

    print(f"✓ Training data saved to: {train_file}")
    print("  Format: JSONL with chat messages")

    # Show how chat template works
    print("\nDemonstrating chat template formatting:")
    messages = sample_data[0]["messages"]
    formatted = format_chat_template(messages, tokenizer)
    print(f"  Original: {messages[0]['content']}")
    print(f"  Formatted length: {len(formatted)} chars")
    print(f"  Template applied: ✓ (supports Llama, Mistral, Qwen, etc.)")

    # =========================================================================
    # Step 4: Configure LoRA for Fine-Tuning
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Configure LoRA for Parameter-Efficient Fine-Tuning")
    print("=" * 70)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
    )

    print("✓ LoRA configured!")
    print(f"  Rank: {model.lora_config['r']}")
    print(f"  Alpha: {model.lora_config['lora_alpha']}")
    print(f"  Target modules: {len(model.lora_config['target_modules'])}")

    # =========================================================================
    # Step 5: Train the Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Train the Model")
    print("=" * 70)

    print("\nFor actual training, run MLX-LM command:")
    print("-" * 70)
    print("mlx_lm.lora \\")
    print("    --model mlx-community/Llama-3.2-1B-Instruct-4bit \\")
    print("    --train \\")
    print(f"    --data {train_file} \\")
    print("    --iters 100 \\")
    print("    --learning-rate 2e-4 \\")
    print("    --lora-layers 16 \\")
    print("    --batch-size 4 \\")
    print("    --adapter-path ./adapters")
    print("-" * 70)

    # Get recommended training config
    config = get_training_config(
        num_train_epochs=3,
        learning_rate=2e-4,
        batch_size=4,
        lora_r=16,
        lora_alpha=32,
    )

    print("\n✓ Training configuration ready:")
    print(f"  Epochs: {config['num_train_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Batch size: {config['batch_size']}")

    print("\nNote: Training happens via mlx_lm.lora command or custom training loop")
    print("      After training, adapters are saved in ./adapters/ directory")

    # =========================================================================
    # Step 6: Save Model in HuggingFace Format
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Save Fine-Tuned Model in Standard HuggingFace Format")
    print("=" * 70)

    print("\nAfter training, save your model so ANYONE can use it:")
    print("  - Not MLX-specific format")
    print("  - Standard HuggingFace transformers format")
    print("  - Can be loaded with transformers, vLLM, etc.")
    print("  - Can be shared on HuggingFace Hub")

    print("\nExample save commands:")
    print("-" * 70)
    print("# Save locally")
    print("save_model_hf_format(model, tokenizer, './my-finetuned-model')")
    print()
    print("# Save and upload to HuggingFace Hub")
    print("save_model_hf_format(")
    print("    model, tokenizer,")
    print("    './my-finetuned-model',")
    print("    push_to_hub=True,")
    print("    repo_id='username/my-awesome-model'")
    print(")")
    print("-" * 70)

    print("\n✓ Model can be saved in standard HF format")
    print("  Others can use: transformers.AutoModel.from_pretrained('your-model')")

    # =========================================================================
    # Step 7: Export to GGUF (Optional)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Export to GGUF for llama.cpp, Ollama, etc. (Optional)")
    print("=" * 70)

    print("\nGGUF export enables:")
    print("  - Use with llama.cpp")
    print("  - Use with Ollama")
    print("  - Use with GPT4All")
    print("  - CPU-optimized inference")

    print("\nExample export command:")
    print("-" * 70)
    print("export_to_gguf(")
    print("    './my-finetuned-model',")
    print("    output_path='model-q4.gguf',")
    print("    quantization='q4_k_m'  # or 'q5_k_m', 'q8_0', etc.")
    print(")")
    print("-" * 70)

    print("\n✓ GGUF export supported for maximum compatibility")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE!")
    print("=" * 70)

    print("\nWhat You Can Do:")
    print("  ✓ Load ANY HuggingFace model (not just mlx-community)")
    print("  ✓ Use load_dataset() from HuggingFace datasets")
    print("  ✓ Apply chat templates for different LLMs")
    print("  ✓ Fine-tune with LoRA/QLoRA")
    print("  ✓ Save in standard HuggingFace format")
    print("  ✓ Export to GGUF for deployment")
    print("  ✓ Share on HuggingFace Hub")
    print("  ✓ Use in Jupyter notebooks")

    print("\nJust like Unsloth, but for Apple Silicon!")

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Load your favorite HuggingFace model")
    print("2. Load a real dataset (thousands of examples)")
    print("3. Train with: mlx_lm.lora --model ... --train --data ...")
    print("4. Save in HF format and/or export to GGUF")
    print("5. Share your fine-tuned model with the world!")


if __name__ == "__main__":
    main()
