"""
Calibration data utilities for REAM/REAP MoE compression.

This module provides:
- Dataset registry with commonly used calibration datasets
- Calibration batch creation utilities
- Support for various datasets (C4, code, math, etc.)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class CalibrationBatch:
    """
    Generic container for calibration batches.

    Attributes:
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class TextDataset(Dataset):
    """
    Simple in-memory text dataset for calibration.

    For production use with large datasets, consider using
    HuggingFace datasets with streaming.
    """

    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class DatasetRegistry:
    """
    Registry of calibration datasets with factory functions.

    Supported datasets:
    - c4: General web text (C4 corpus)
    - code: Code corpus (using instruction tuning datasets)
    - math: Math instruction datasets
    - writing: Creative writing prompts
    - custom: User-provided text or dataset
    """

    _datasets: Dict[str, Callable[..., Iterable[str]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a dataset factory function."""

        def decorator(func: Callable[..., Iterable[str]]) -> Callable:
            cls._datasets[name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable[..., Iterable[str]] | None:
        """Get a dataset factory by name."""
        return cls._datasets.get(name)

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names."""
        return sorted(cls._datasets.keys())


# Register built-in datasets


@DatasetRegistry.register("c4")
def _load_c4(
    samples: int = 1000,
    split: str = "train",
    streaming: bool = True,
) -> Iterable[str]:
    """
    Load C4 (Colossal Clean Crawled Corpus) dataset.

    Args:
        samples: Number of samples to load
        split: Dataset split ("train", "validation")
        streaming: Whether to use streaming mode

    Returns:
        Iterable of text samples
    """
    try:
        import datasets
    except ImportError:
        logger.warning("datasets library not installed, using fallback texts")
        return _fallback_texts(samples)

    try:
        logger.info(f"Loading C4 dataset ({samples} samples, streaming={streaming})")
        ds = datasets.load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00000-of-01024.json.gz" if streaming else None,
            split=split,
            streaming=streaming,
        )

        def text_generator():
            count = 0
            for example in ds:
                if count >= samples:
                    break
                yield example["text"]
                count += 1

        return text_generator()

    except Exception as e:
        logger.warning(f"Failed to load C4 dataset: {e}, using fallback")
        return _fallback_texts(samples)


@DatasetRegistry.register("code")
def _load_code(
    samples: int = 500,
    streaming: bool = True,
) -> Iterable[str]:
    """
    Load code instruction dataset.

    Uses a mix of code instruction datasets for calibration.

    Args:
        samples: Number of samples to load
        streaming: Whether to use streaming mode

    Returns:
        Iterable of code-related text samples
    """
    try:
        import datasets
    except ImportError:
        return _fallback_code_texts(samples)

    try:
        logger.info(f"Loading code dataset ({samples} samples, streaming={streaming})")
        ds = datasets.load_dataset(
            "theblackcat102/evol-codealpaca-v1",
            split="train",
            streaming=streaming,
        )

        def text_generator():
            count = 0
            for example in ds:
                if count >= samples:
                    break
                # Use instruction + response as calibration text
                text = example.get("instruction", "") + " " + example.get("output", "")
                yield text.strip()
                count += 1

        return text_generator()

    except Exception as e:
        logger.warning(f"Failed to load code dataset: {e}, using fallback")
        return _fallback_code_texts(samples)


@DatasetRegistry.register("math")
def _load_math(
    samples: int = 500,
    streaming: bool = True,
) -> Iterable[str]:
    """
    Load math instruction dataset.

    Args:
        samples: Number of samples to load
        streaming: Whether to use streaming mode

    Returns:
        Iterable of math-related text samples
    """
    try:
        import datasets
    except ImportError:
        return _fallback_math_texts(samples)

    try:
        logger.info(f"Loading math dataset ({samples} samples, streaming={streaming})")
        ds = datasets.load_dataset(
            "allenai/tulu-3-sft-personas-math",
            split="train",
            streaming=streaming,
        )

        def text_generator():
            count = 0
            for example in ds:
                if count >= samples:
                    break
                text = example.get("text", example.get("prompt", ""))
                yield text.strip()
                count += 1

        return text_generator()

    except Exception as e:
        logger.warning(f"Failed to load math dataset: {e}, using fallback")
        return _fallback_math_texts(samples)


@DatasetRegistry.register("writing")
def _load_writing(
    samples: int = 300,
    streaming: bool = True,
) -> Iterable[str]:
    """
    Load creative writing prompts dataset.

    Args:
        samples: Number of samples to load
        streaming: Whether to use streaming mode

    Returns:
        Iterable of writing-related text samples
    """
    try:
        import datasets
    except ImportError:
        return _fallback_writing_texts(samples)

    try:
        logger.info(f"Loading writing dataset ({samples} samples, streaming={streaming})")
        ds = datasets.load_dataset(
            "euclaise/WritingPrompts_curated",
            split="train",
            streaming=streaming,
        )

        def text_generator():
            count = 0
            for example in ds:
                if count >= samples:
                    break
                text = example.get("text", example.get("prompt", ""))
                yield text.strip()
                count += 1

        return text_generator()

    except Exception as e:
        logger.warning(f"Failed to load writing dataset: {e}, using fallback")
        return _fallback_writing_texts(samples)


@DatasetRegistry.register("combined")
def _load_combined(
    samples_per_category: int = 250,
    streaming: bool = True,
) -> Iterable[str]:
    """
    Load a combined dataset with multiple categories.

    Args:
        samples_per_category: Number of samples per category
        streaming: Whether to use streaming mode

    Returns:
        Iterable of text samples from all categories
    """
    categories = ["c4", "code", "math", "writing"]
    total_samples = samples_per_category * len(categories)

    def text_generator():
        for category in categories:
            factory = DatasetRegistry.get(category)
            if factory is None:
                logger.warning(f"Category {category} not found, skipping")
                continue
            try:
                for text in factory(samples=samples_per_category, streaming=streaming):
                    yield text
            except Exception as e:
                logger.warning(f"Failed to load category {category}: {e}")

    return text_generator()


def _fallback_texts(samples: int) -> List[str]:
    """Fallback texts when datasets library is not available."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing deals with the interaction between computers and human language.",
        "Deep learning models are trained using large amounts of data.",
        "The transformer architecture revolutionized natural language processing.",
        "Mixture of Experts models use specialized sub-networks called experts.",
        "Expert routing determines which experts process each token.",
        "Model compression reduces the size and computational cost of neural networks.",
        "Pruning removes unnecessary parameters from trained models.",
        "Knowledge distillation transfers knowledge from a large model to a smaller one.",
    ]
    return (texts * ((samples // len(texts)) + 1))[:samples]


def _fallback_code_texts(samples: int) -> List[str]:
    """Fallback code texts when datasets library is not available."""
    texts = [
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "class Node: def __init__(self, val): self.val = val; self.next = None",
        "for i in range(len(arr)): if arr[i] > max_val: max_val = arr[i]",
        "async def fetch_data(url): async with aiohttp.ClientSession() as session: response = await session.get(url)",
        "from typing import List, Optional, Dict, Any",
        "def binary_search(arr, target): left, right = 0, len(arr) - 1",
        "@dataclass class User: id: int; name: str; email: str",
        "import torch; import torch.nn as nn; import torch.nn.functional as F",
        "try: result = dangerous_operation(); except Exception as e: logger.error(f'Error: {e}')",
        "with open('file.txt', 'r') as f: content = f.read()",
    ]
    return (texts * ((samples // len(texts)) + 1))[:samples]


def _fallback_math_texts(samples: int) -> List[str]:
    """Fallback math texts when datasets library is not available."""
    texts = [
        "Solve for x: 2x + 5 = 15, therefore x = 5",
        "The derivative of x^2 is 2x",
        "The integral of 1/x from 1 to infinity is ln(infinity) - ln(1)",
        "Pythagorean theorem: a^2 + b^2 = c^2 for right triangles",
        "The sum of angles in a triangle equals 180 degrees or π radians",
        "Bayes theorem: P(A|B) = P(B|A) * P(A) / P(B)",
        "Euler's identity: e^(iπ) + 1 = 0",
        "The limit of sin(x)/x as x approaches 0 is 1",
        "Standard deviation = sqrt(sum((x - mean)^2) / N)",
        "Matrix multiplication: (AB)_{ij} = sum_k A_{ik} * B_{kj}",
    ]
    return (texts * ((samples // len(texts)) + 1))[:samples]


def _fallback_writing_texts(samples: int) -> List[str]:
    """Fallback writing texts when datasets library is not available."""
    texts = [
        "Write a story about a time traveler who accidentally changes history.",
        "Describe the experience of waking up in a foreign city for the first time.",
        "Compose a poem about the changing seasons from the perspective of a tree.",
        "Create a dialogue between two old friends who haven't seen each other in decades.",
        "Write about a character who discovers they can see the future in their dreams.",
        "Describe the feeling of accomplishment after completing a long-term project.",
        "Write a scene where two characters from different time periods meet.",
        "Create a mystery that takes place in a small coastal town.",
        "Write about someone learning a new skill and overcoming initial frustration.",
        "Describe the atmosphere of a bustling marketplace at sunset.",
    ]
    return (texts * ((samples // len(texts)) + 1))[:samples]


def build_calibration_batches(
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str] | str | List[str],
    max_seq_len: int = 512,
    batch_size: int = 4,
) -> Iterable[CalibrationBatch]:
    """
    Build calibration batches from texts.

    Args:
        tokenizer: Tokenizer to use for encoding
        texts: Text samples to encode. Can be:
            - Iterable of strings (already loaded texts)
            - String name of registered dataset
            - List of strings
        max_seq_len: Maximum sequence length
        batch_size: Batch size for calibration

    Returns:
        Iterable of CalibrationBatch objects

    Examples:
        >>> # Use texts directly
        >>> batches = build_calibration_batches(tokenizer, ["text1", "text2"])
        >>> # Use registered dataset
        >>> batches = build_calibration_batches(tokenizer, "c4")
        >>> # Use combined dataset with custom samples
        >>> batches = build_calibration_batches(tokenizer, "combined")
    """
    # Handle dataset name
    if isinstance(texts, str):
        factory = DatasetRegistry.get(texts)
        if factory is None:
            logger.warning(f"Dataset '{texts}' not found, using fallback")
            texts = _fallback_texts(1000)
        else:
            texts = factory(samples=1000)

    # Handle list/iterable of texts
    if isinstance(texts, list):
        text_list = texts
    elif hasattr(texts, "__iter__"):
        text_list = list(texts)
    else:
        raise ValueError(f"Invalid texts type: {type(texts)}")

    dataset = TextDataset(text_list)

    def collate(batch_texts: List[str]) -> CalibrationBatch:
        enc = tokenizer(
            batch_texts,
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return CalibrationBatch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    for batch in loader:
        yield batch


def get_dataset_factory(name: str) -> Callable[..., Iterable[str]] | None:
    """
    Get a dataset factory function by name.

    Args:
        name: Dataset name (e.g., "c4", "code", "math", "writing", "combined")

    Returns:
        Dataset factory function or None if not found
    """
    return DatasetRegistry.get(name)


def list_available_datasets() -> List[str]:
    """List all available dataset names."""
    return DatasetRegistry.list_datasets()
