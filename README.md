# Hox

**Hox** is a lightweight ONNX-based LLM inference engine for running supported language models locally.

It provides simple text generation and streaming generation through an easy-to-use Python API.

## Features

- ONNX-based LLM inference
- Streaming token generation
- Standard text generation
- Sampling controls such as temperature and top-k
- Repetition penalty
- Configurable maximum output tokens
- Hugging Face Hub model support
- Designed for models with **KV-cache support**

## Installation

Install Hox from the project source:

```bash
git clone https://github.com/Gananath/hox.git
cd hox
pip install .
```

Or install the package directly:

```bash
pip install git+https://github.com/Gananath/hox.git
```

### Requirements

- Python `>= 3.8`

The package depends on:

- Jinja2 `>= 3.1.6`
- ONNX `>= 1.19.1`
- ONNX Runtime `>= 1.23.1`
- tokenizers `>= 0.22.2`
- huggingface_hub `>= 1.25.1`

## Quick Start

Import `OnnxLLM` and load a supported model from Hugging Face:

```python
from hox import OnnxLLM

model = OnnxLLM("LiquidAI/LFM2.5-230M-ONNX", "fp16")
```

The second argument specifies the model variant/precision.

### Streaming Generation

Tokens can be generated incrementally using `stream()`:

```python
from hox import OnnxLLM

model = OnnxLLM("LiquidAI/LFM2.5-230M-ONNX", "fp16")

for token in model.stream("What is C. elegans?"):
    print(token, end="", flush=True)
```

### Text Generation

For regular generation, use `generate()`:

```python
from hox import OnnxLLM

model = OnnxLLM("LiquidAI/LFM2.5-230M-ONNX", "fp16")

response = model.generate(
    "What is C. elegans?",
    do_sample=True,
    temperature=0.2,
    top_k=80,
    repetition_penalty=1.05,
    max_new_tokens=512,
)

print(response)
```

## Tested Models

The following models have been tested with Hox:

| # | Repo / Name |
|---:|---|
| 1 | `gemma-3-270m-it-ONNX` |
| 2 | `LFM2.5-1.2B-Instruct-ONNX` |
| 3 | `LFM2.5-230M-ONNX` |
| 4 | `LFM2-350M-ONNX` |
| 5 | `LFM2.5-1.2B-Thinking-ONNX` |
| 6 | `LFM2.5-350M-ONNX` |
| 7 | `Qwen3-0.6B-ONNX` |

## Current Limitations

Hox currently supports **only models that use a KV cache** for autoregressive generation.

The following are **not currently supported**:

- Vision-language models (VLMs)
- Multimodal models
- Models that do not support/use KV-cache based generation


## API

### `OnnxLLM`

Load a model with:

```python
model = OnnxLLM("LiquidAI/LFM2.5-230M-ONNX", "fp16")
```

### `stream()`

Generate text incrementally:

```python
for token in model.stream("Your prompt here"):
    print(token, end="", flush=True)
```

### `generate()`

Generate a complete response:

```python
response = model.generate(
    "Your prompt here",
    do_sample=True,
    temperature=0.2,
    top_k=80,
    repetition_penalty=1.05,
    max_new_tokens=512,
)

print(response)
```

### `Tool` usage

Define tools with `@Tool` and pass them to the chat template:

```python
from hox import OnnxLLM, Tool

@Tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [multiply]

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 25 * 4?"},
]

prompt = model.template.apply_chat_template(
    messages,
    tools=tools,
    add_generation_prompt=True,
)
```

## Project Status

Hox is currently under active development. Model compatibility is being expanded, with the current focus on efficient KV-cache based ONNX language model inference.

Contributions, bug reports, and model compatibility testing are welcome.

