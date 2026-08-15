#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import logging
import os

import numpy as np
from .base import ONNXBase
from .chat_template import ChatTemplate
from tokenizers import Tokenizer
from .utils import select_token



# In[2]:


class OnnxLLM(ONNXBase):
    ONNX_TYPE_TO_NUMPY = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }

    def __init__(
        self,
        repo_id: str = "LiquidAI/LFM2.5-230M-ONNX",
        precision: str = "fp16",
        required_files=None,
        save_dir=None,
    ):
        super().__init__(
            repo_id=repo_id,
            precision=precision,
            required_files=required_files,
            save_dir=save_dir,
        )

        self._load_config()
        self._load_tokenizer()
        self.load_session()

        self.template = ChatTemplate(
            self._save_dir,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_config(self):
        config_path = os.path.join(
            self._save_dir,
            "config.json",
        )

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.eos_token_id = self.config["eos_token_id"]
        self.bos_token_id = self.config.get("bos_token_id")
        self.pad_token_id = self.config.get("pad_token_id")
        self.vocab_size = self.config.get("vocab_size")

    def _load_tokenizer(self):
        tokenizer_path = os.path.join(
            self._save_dir,
            "tokenizer.json",
        )

        self.tokenizer = Tokenizer.from_file(
            tokenizer_path,
        )

    def _get_onnx_path(self) -> str:
        return os.path.join(
            self._save_dir,
            "onnx",
            f"model_{self.precision}.onnx",
        )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def initialize_cache(self) -> dict:
        skip_inputs = {
            "input_ids",
            "inputs_embeds",
            "attention_mask",
            "position_ids",
        }

        cache = {}

        for inp in self.session.get_inputs():
            if inp.name in skip_inputs:
                continue

            shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]

            for i, dim in enumerate(inp.shape):
                if isinstance(dim, str) and "sequence" in dim.lower():
                    shape[i] = 0

            dtype = self.ONNX_TYPE_TO_NUMPY.get(
                inp.type,
                np.float32,
            )

            cache[inp.name] = np.zeros(
                shape,
                dtype=dtype,
            )

        return cache

    def update_cache(
        self,
        cache: dict,
        outputs: list,
    ):
        for i, out_info in enumerate(
            self.output_infos[1:],
            1,
        ):
            name = out_info.name

            if "present_conv" in name:
                cache_name = name.replace(
                    "present_conv",
                    "past_conv",
                )

            elif "present." in name:
                cache_name = name.replace(
                    "present.",
                    "past_key_values.",
                )

            else:
                continue

            if cache_name in cache:
                cache[cache_name] = outputs[i]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_tokens(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ):
        """
        Generate tokens from the model.

        This is shared by generate() and stream().
        """

        encoding = self.tokenizer.encode(prompt)

        input_ids = np.array(
            [encoding.ids],
            dtype=np.int64,
        )

        prompt_tokens = list(encoding.ids)

        generated_tokens = []

        cache = self.initialize_cache()

        seq_len = input_ids.shape[1]
        cur_len = seq_len

        position_ids = np.arange(
            seq_len,
            dtype=np.int64,
        ).reshape(1, -1)

        for step in range(max_new_tokens):
            # ----------------------------------------------------------
            # Prepare inputs
            # ----------------------------------------------------------

            if step == 0:
                ids = input_ids
                pos = position_ids
            else:
                ids = np.array(
                    [[generated_tokens[-1]]],
                    dtype=np.int64,
                )

                pos = np.array(
                    [[cur_len - 1]],
                    dtype=np.int64,
                )

            attention_mask = np.ones(
                (1, cur_len),
                dtype=np.int64,
            )

            feed = {
                "input_ids": ids,
                "attention_mask": attention_mask,
            }

            if "position_ids" in self.input_names:
                feed["position_ids"] = pos

            feed.update(cache)

            # ----------------------------------------------------------
            # ONNX inference
            # ----------------------------------------------------------

            outputs = self.session.run(
                None,
                feed,
            )

            logits = outputs[0][0, -1]

            # ----------------------------------------------------------
            # Token selection
            # ----------------------------------------------------------

            next_token = select_token(
                logits=logits,
                token_ids=prompt_tokens + generated_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

            generated_tokens.append(next_token)

            # ----------------------------------------------------------
            # Update KV cache
            # ----------------------------------------------------------

            self.update_cache(
                cache,
                outputs,
            )

            cur_len += 1

            # ----------------------------------------------------------
            # Stop condition
            # ----------------------------------------------------------

            if next_token == self.eos_token_id:
                break

            yield next_token

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Generate the complete response."""

        generated_tokens = list(
            self._generate_tokens(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        )

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=False,
        )

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ):
        """Stream generated tokens."""

        for token_id in self._generate_tokens(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        ):
            yield self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )


# In[6]:


if __name__ == "__main__":
    model = LiquideLLMModel("LiquidAI/LFM2.5-230M-ONNX", "fp16")

    for token in model.stream("What is C. elegans?"):
        print(token, end="", flush=True)

    response = model.generate(
        "What is C. elegans?",
        do_sample=True,
        temperature=0.2,
        top_k=80,
        repetition_penalty=1.05,
        max_new_tokens=512,
    )
    print(response)


    # In[7]:


    messages = [
        {
            "role": "system",
            "content": "You are helpful. You have access to various tools"
        },
        {"role": "user", "content": "What is the time like in london"},
    ]

    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The name of the city."}
                },
                "required": ["city"],
            },
        },
        {
            "name": "get_time",
            "description": "Get the current local time for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The name of the city."}
                },
                "required": ["city"],
            },
        },
    ]

    model.template.apply_chat_template(messages, tools)


# In[ ]:




