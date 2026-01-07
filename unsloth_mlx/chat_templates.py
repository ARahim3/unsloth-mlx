"""
Chat Templates and Dataset Formatting for Unsloth-MLX

This module provides Unsloth-compatible dataset formatting utilities,
converting various dataset formats to mlx-lm compatible formats.

Supported input formats:
- Alpaca: {"instruction": "...", "input": "...", "output": "..."}
- ShareGPT: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
- ChatML: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
- Text: {"text": "..."}
- Completions: {"prompt": "...", "completion": "..."}

Output formats (mlx-lm compatible):
- text: {"text": "..."}
- chat: {"messages": [...]}
- completions: {"prompt": "...", "completion": "..."}
"""

from typing import Any, Dict, List, Optional, Callable, Union
from datasets import Dataset


# Default Alpaca prompt template
ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


def detect_dataset_format(sample: Dict[str, Any]) -> str:
    """
    Detect the format of a dataset sample.

    Args:
        sample: A single sample from the dataset

    Returns:
        Format string: 'alpaca', 'sharegpt', 'chatml', 'text', 'completions', or 'unknown'
    """
    keys = set(sample.keys())

    # Check for text format (simplest)
    if 'text' in keys:
        return 'text'

    # Check for ChatML format (messages with role/content)
    if 'messages' in keys:
        messages = sample['messages']
        if isinstance(messages, list) and len(messages) > 0:
            if isinstance(messages[0], dict) and 'role' in messages[0]:
                return 'chatml'

    # Check for ShareGPT format (conversations with from/value)
    if 'conversations' in keys:
        convos = sample['conversations']
        if isinstance(convos, list) and len(convos) > 0:
            if isinstance(convos[0], dict) and 'from' in convos[0]:
                return 'sharegpt'

    # Check for completions format
    if 'prompt' in keys and 'completion' in keys:
        return 'completions'

    # Check for Alpaca format
    if 'instruction' in keys and 'output' in keys:
        return 'alpaca'

    return 'unknown'


def standardize_sharegpt(dataset: Dataset) -> Dataset:
    """
    Convert ShareGPT format to ChatML format.

    ShareGPT uses {"from": "human/gpt", "value": "..."}
    ChatML uses {"role": "user/assistant", "content": "..."}

    Args:
        dataset: Dataset with ShareGPT format conversations

    Returns:
        Dataset with ChatML format messages
    """
    role_mapping = {
        'human': 'user',
        'user': 'user',
        'gpt': 'assistant',
        'assistant': 'assistant',
        'system': 'system',
    }

    def convert_sample(sample):
        if 'conversations' not in sample:
            return sample

        messages = []
        for turn in sample['conversations']:
            role = role_mapping.get(turn.get('from', '').lower(), 'user')
            content = turn.get('value', '')
            messages.append({'role': role, 'content': content})

        return {'messages': messages}

    return dataset.map(convert_sample)


def alpaca_to_text(
    sample: Dict[str, Any],
    template: Optional[str] = None,
) -> str:
    """
    Convert Alpaca format sample to text.

    Args:
        sample: Alpaca format sample with instruction, input, output
        template: Optional custom template string

    Returns:
        Formatted text string
    """
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    output = sample.get('output', '')

    if template:
        return template.format(
            instruction=instruction,
            input=input_text,
            output=output
        )

    # Use appropriate template based on whether input is provided
    if input_text.strip():
        return ALPACA_TEMPLATE.format(
            instruction=instruction,
            input=input_text,
            output=output
        )
    else:
        return ALPACA_TEMPLATE_NO_INPUT.format(
            instruction=instruction,
            output=output
        )


def apply_chat_template_to_sample(
    sample: Dict[str, Any],
    tokenizer: Any,
    add_generation_prompt: bool = False,
) -> str:
    """
    Apply tokenizer's chat template to a sample.

    Args:
        sample: Sample with 'messages' field (ChatML format)
        tokenizer: Tokenizer with apply_chat_template method
        add_generation_prompt: Whether to add generation prompt

    Returns:
        Formatted text string
    """
    messages = sample.get('messages', [])

    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
    else:
        # Fallback: simple formatting
        text_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            text_parts.append(f"{role}: {content}")
        return '\n'.join(text_parts)


def convert_to_mlx_format(
    dataset: Dataset,
    tokenizer: Any,
    output_format: str = 'text',
    alpaca_template: Optional[str] = None,
) -> Dataset:
    """
    Convert any supported dataset format to mlx-lm compatible format.

    This is the main function for dataset conversion, similar to Unsloth's
    formatting workflow.

    Args:
        dataset: Input dataset in any supported format
        tokenizer: Tokenizer (used for chat template if available)
        output_format: Target format ('text', 'chat', 'completions')
        alpaca_template: Custom template for Alpaca format conversion

    Returns:
        Dataset in mlx-lm compatible format

    Example:
        >>> from unsloth_mlx.chat_templates import convert_to_mlx_format
        >>> dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")
        >>> dataset = convert_to_mlx_format(dataset, tokenizer)
        >>> # Now dataset has 'text' field compatible with mlx-lm
    """
    if len(dataset) == 0:
        return dataset

    # Detect input format from first sample
    input_format = detect_dataset_format(dataset[0])
    print(f"Detected dataset format: {input_format}")

    if input_format == 'unknown':
        print(f"Warning: Unknown dataset format. Fields: {list(dataset[0].keys())}")
        print("Attempting to use raw sample...")

    def convert_sample(sample):
        # Already in target format?
        if input_format == output_format:
            return sample
        if input_format == 'text' and output_format == 'text':
            return sample

        # Convert based on input format
        if input_format == 'alpaca':
            if output_format == 'text':
                text = alpaca_to_text(sample, alpaca_template)
                return {'text': text}
            elif output_format == 'completions':
                instruction = sample.get('instruction', '')
                input_text = sample.get('input', '')
                prompt = f"{instruction}\n{input_text}".strip() if input_text else instruction
                return {'prompt': prompt, 'completion': sample.get('output', '')}
            elif output_format == 'chat':
                # Convert to messages format
                messages = [
                    {'role': 'user', 'content': f"{sample.get('instruction', '')}\n{sample.get('input', '')}".strip()},
                    {'role': 'assistant', 'content': sample.get('output', '')}
                ]
                return {'messages': messages}

        elif input_format == 'sharegpt':
            # First convert to ChatML
            messages = []
            role_mapping = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
            for turn in sample.get('conversations', []):
                role = role_mapping.get(turn.get('from', '').lower(), 'user')
                messages.append({'role': role, 'content': turn.get('value', '')})

            if output_format == 'chat':
                return {'messages': messages}
            elif output_format == 'text':
                text = apply_chat_template_to_sample({'messages': messages}, tokenizer)
                return {'text': text}

        elif input_format == 'chatml':
            if output_format == 'chat':
                return sample  # Already in chat format
            elif output_format == 'text':
                text = apply_chat_template_to_sample(sample, tokenizer)
                return {'text': text}

        elif input_format == 'completions':
            if output_format == 'completions':
                return sample
            elif output_format == 'text':
                return {'text': f"{sample.get('prompt', '')}\n{sample.get('completion', '')}"}

        elif input_format == 'text':
            return sample

        # Fallback for unknown format - try to create text
        if output_format == 'text':
            # Try common field names
            for field in ['text', 'content', 'output', 'response', 'completion']:
                if field in sample:
                    return {'text': sample[field]}
            # Last resort: stringify the sample
            import json
            return {'text': json.dumps(sample)}

        return sample

    converted = dataset.map(convert_sample)

    # Verify conversion
    if len(converted) > 0:
        result_format = detect_dataset_format(converted[0])
        print(f"Output format: {result_format}")
        if output_format == 'text' and 'text' not in converted[0]:
            print(f"Warning: Conversion may have failed. Sample keys: {list(converted[0].keys())}")

    return converted


def get_formatting_func(
    tokenizer: Any,
    dataset_format: str = 'auto',
    alpaca_template: Optional[str] = None,
) -> Callable:
    """
    Get a formatting function for use with SFTTrainer.

    This returns a function that can be passed to SFTTrainer's formatting_func
    parameter to automatically convert samples to the text format.

    Args:
        tokenizer: Tokenizer instance
        dataset_format: Expected format ('auto', 'alpaca', 'sharegpt', 'chatml')
        alpaca_template: Custom template for Alpaca format

    Returns:
        Formatting function that takes a sample and returns formatted text

    Example:
        >>> formatting_func = get_formatting_func(tokenizer)
        >>> trainer = SFTTrainer(
        ...     model=model,
        ...     train_dataset=dataset,
        ...     formatting_func=formatting_func,
        ...     ...
        ... )
    """
    def formatting_func(sample: Dict[str, Any]) -> str:
        # Detect format if auto
        fmt = dataset_format
        if fmt == 'auto':
            fmt = detect_dataset_format(sample)

        # Convert based on format
        if fmt == 'text':
            return sample.get('text', '')

        elif fmt == 'alpaca':
            return alpaca_to_text(sample, alpaca_template)

        elif fmt == 'sharegpt':
            # Convert to ChatML first
            messages = []
            role_mapping = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
            for turn in sample.get('conversations', []):
                role = role_mapping.get(turn.get('from', '').lower(), 'user')
                messages.append({'role': role, 'content': turn.get('value', '')})
            return apply_chat_template_to_sample({'messages': messages}, tokenizer)

        elif fmt == 'chatml':
            return apply_chat_template_to_sample(sample, tokenizer)

        elif fmt == 'completions':
            return f"{sample.get('prompt', '')}\n{sample.get('completion', '')}"

        else:
            # Unknown format - try to extract text
            for field in ['text', 'content', 'output', 'response']:
                if field in sample:
                    return sample[field]
            return str(sample)

    return formatting_func


# Convenience exports matching Unsloth API
__all__ = [
    'detect_dataset_format',
    'standardize_sharegpt',
    'convert_to_mlx_format',
    'get_formatting_func',
    'apply_chat_template_to_sample',
    'alpaca_to_text',
    'ALPACA_TEMPLATE',
    'ALPACA_TEMPLATE_NO_INPUT',
]
