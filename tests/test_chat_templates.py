"""
Tests for chat_templates module - dataset format detection and conversion.

These tests ensure that various dataset formats are correctly detected and
converted to mlx-lm compatible formats.
"""

import pytest
from datasets import Dataset

from unsloth_mlx.chat_templates import (
    detect_dataset_format,
    standardize_sharegpt,
    convert_to_mlx_format,
    get_formatting_func,
    alpaca_to_text,
    ALPACA_TEMPLATE,
    ALPACA_TEMPLATE_NO_INPUT,
)


class TestDetectDatasetFormat:
    """Test dataset format detection."""

    def test_detect_text_format(self):
        sample = {"text": "Hello world"}
        assert detect_dataset_format(sample) == "text"

    def test_detect_alpaca_format(self):
        sample = {
            "instruction": "Translate to French",
            "input": "Hello",
            "output": "Bonjour"
        }
        assert detect_dataset_format(sample) == "alpaca"

    def test_detect_alpaca_without_input(self):
        sample = {
            "instruction": "Tell me a joke",
            "output": "Why did the chicken..."
        }
        assert detect_dataset_format(sample) == "alpaca"

    def test_detect_sharegpt_format(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there!"}
            ]
        }
        assert detect_dataset_format(sample) == "sharegpt"

    def test_detect_chatml_format(self):
        sample = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ]
        }
        assert detect_dataset_format(sample) == "chatml"

    def test_detect_completions_format(self):
        sample = {
            "prompt": "What is 2+2?",
            "completion": "4"
        }
        assert detect_dataset_format(sample) == "completions"

    def test_detect_unknown_format(self):
        sample = {"foo": "bar", "baz": 123}
        assert detect_dataset_format(sample) == "unknown"


class TestAlpacaToText:
    """Test Alpaca format to text conversion."""

    def test_alpaca_with_input(self):
        sample = {
            "instruction": "Translate to French",
            "input": "Hello",
            "output": "Bonjour"
        }
        text = alpaca_to_text(sample)

        assert "Translate to French" in text
        assert "Hello" in text
        assert "Bonjour" in text
        assert "### Instruction:" in text
        assert "### Input:" in text
        assert "### Response:" in text

    def test_alpaca_without_input(self):
        sample = {
            "instruction": "Tell me a joke",
            "input": "",
            "output": "Why did the chicken cross the road?"
        }
        text = alpaca_to_text(sample)

        assert "Tell me a joke" in text
        assert "Why did the chicken" in text
        # Should NOT have ### Input: section when input is empty
        assert "### Input:" not in text

    def test_alpaca_custom_template(self):
        sample = {
            "instruction": "Do something",
            "input": "with this",
            "output": "done"
        }
        template = "Q: {instruction} {input}\nA: {output}"
        text = alpaca_to_text(sample, template=template)

        assert text == "Q: Do something with this\nA: done"


class TestStandardizeSharegpt:
    """Test ShareGPT to ChatML conversion."""

    def test_sharegpt_to_chatml(self):
        data = {
            "conversations": [
                [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there!"}
                ]
            ]
        }
        dataset = Dataset.from_dict(data)
        converted = standardize_sharegpt(dataset)

        assert "messages" in converted[0]
        messages = converted[0]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    def test_sharegpt_with_system(self):
        data = {
            "conversations": [
                [
                    {"from": "system", "value": "You are helpful"},
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi!"}
                ]
            ]
        }
        dataset = Dataset.from_dict(data)
        converted = standardize_sharegpt(dataset)

        messages = converted[0]["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "system"


class TestConvertToMlxFormat:
    """Test full dataset conversion to mlx-lm format."""

    def test_convert_alpaca_to_text(self):
        """Test that yahma/alpaca-cleaned style data is converted properly."""
        data = {
            "instruction": ["Give tips for health", "Translate hello"],
            "input": ["", "to French"],
            "output": ["Eat well, sleep well", "Bonjour"]
        }
        dataset = Dataset.from_dict(data)

        # Mock tokenizer
        class MockTokenizer:
            pass

        converted = convert_to_mlx_format(dataset, MockTokenizer(), output_format='text')

        # Should have 'text' field
        assert "text" in converted[0]
        assert "Eat well" in converted[0]["text"]
        assert "Give tips" in converted[0]["text"]

    def test_convert_sharegpt_to_chat(self):
        """Test ShareGPT to chat format conversion."""
        data = {
            "conversations": [
                [
                    {"from": "human", "value": "What is AI?"},
                    {"from": "gpt", "value": "AI is..."}
                ]
            ]
        }
        dataset = Dataset.from_dict(data)

        class MockTokenizer:
            pass

        converted = convert_to_mlx_format(dataset, MockTokenizer(), output_format='chat')

        assert "messages" in converted[0]
        assert converted[0]["messages"][0]["role"] == "user"

    def test_text_passthrough(self):
        """Test that text format passes through unchanged."""
        data = {"text": ["Hello world", "Test sample"]}
        dataset = Dataset.from_dict(data)

        class MockTokenizer:
            pass

        converted = convert_to_mlx_format(dataset, MockTokenizer(), output_format='text')

        assert converted[0]["text"] == "Hello world"


class TestGetFormattingFunc:
    """Test formatting function generation."""

    def test_formatting_func_alpaca(self):
        class MockTokenizer:
            pass

        func = get_formatting_func(MockTokenizer(), dataset_format='alpaca')

        sample = {
            "instruction": "Test instruction",
            "input": "Test input",
            "output": "Test output"
        }

        result = func(sample)
        assert isinstance(result, str)
        assert "Test instruction" in result
        assert "Test output" in result

    def test_formatting_func_text(self):
        class MockTokenizer:
            pass

        func = get_formatting_func(MockTokenizer(), dataset_format='text')

        sample = {"text": "Hello world"}
        result = func(sample)
        assert result == "Hello world"

    def test_formatting_func_auto_detect(self):
        """Test auto-detection in formatting function."""
        class MockTokenizer:
            pass

        func = get_formatting_func(MockTokenizer(), dataset_format='auto')

        # Should detect alpaca format
        alpaca_sample = {
            "instruction": "Do something",
            "input": "",
            "output": "Done"
        }
        result = func(alpaca_sample)
        assert "Do something" in result
        assert "Done" in result


class TestImports:
    """Test that all exports work correctly."""

    def test_imports_from_package(self):
        """Test importing from main package."""
        from unsloth_mlx import (
            detect_dataset_format,
            standardize_sharegpt,
            convert_to_mlx_format,
            get_formatting_func,
            alpaca_to_text,
        )

        # All should be callable
        assert callable(detect_dataset_format)
        assert callable(standardize_sharegpt)
        assert callable(convert_to_mlx_format)
        assert callable(get_formatting_func)
        assert callable(alpaca_to_text)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
