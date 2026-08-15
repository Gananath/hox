#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import html
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from jinja2 import Environment, FileSystemLoader, nodes
from jinja2.ext import Extension
from . import config


# In[2]:


class GenerationExtension(Extension):
    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return nodes.CallBlock(
            self.call_method("_"),
            [],
            [],
            parser.parse_statements(
                ("name:endgeneration",),
                drop_needle=True,
            ),
        )

    _ = staticmethod(lambda caller: caller())


class ChatTemplate:
    def __init__(self, save_dir: str):
        self._save_dir = save_dir

        self.template = None
        self.tokenizer_config: Dict[str, Any] = {}

        self._load_template()
        self._load_tokenizer_config()

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def _json_path(self, filename: str) -> str:
        return os.path.join(self._save_dir, filename)

    def _load_json(
        self,
        filename: str,
        default: Any = None,
    ) -> Any:
        path = self._json_path(filename)

        if not os.path.isfile(path):
            return default

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_tokenizer_config(self) -> None:
        self.tokenizer_config = self._load_json(
            "tokenizer_config.json",
            default={},
        )

    # ------------------------------------------------------------------
    # Jinja helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strftime_now(format_string: str) -> str:
        """
        Hugging Face-compatible strftime_now helper.

        Example:

            {{ strftime_now("%Y-%m-%d") }}

        """
        return datetime.now(timezone.utc).strftime(format_string)

    @staticmethod
    def _raise_exception(message: str) -> None:
        """
        Helper used by some Hugging Face chat templates.

        Templates commonly contain:

            {{ raise_exception("Invalid message") }}
        """
        raise ValueError(message)

    def _register_jinja_helpers(
        self,
        env: Environment,
    ) -> Environment:
        """
        Register helpers expected by Hugging Face chat templates.
        """

        env.globals.update(
            {
                "strftime_now": self._strftime_now,
                "raise_exception": self._raise_exception,
            }
        )

        return env

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def _find_template_file(self) -> str:
        template_files = glob.glob(os.path.join(self._save_dir, "*.jinja"))

        if not template_files:
            raise FileNotFoundError(f"No .jinja template found in {self._save_dir}")

        template_files.sort()

        return template_files[0]

    def _create_environment(self) -> Environment:
        """
        Create a Jinja environment compatible with HF chat templates.
        """

        env = Environment(
            loader=FileSystemLoader(self._save_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
            extensions=[GenerationExtension],
        )

        return self._register_jinja_helpers(env)

    def _load_template(self) -> None:
        template_file = self._find_template_file()

        env = self._create_environment()

        self.template = env.get_template(os.path.basename(template_file))

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _build_template_kwargs(
        self,
        messages,
        tools,
        documents,
        add_generation_prompt,
        continue_final_message,
        **kwargs,
    ):
        template_kwargs = {
            "messages": messages,
            "tools": tools,
            "documents": documents,
            "add_generation_prompt": add_generation_prompt,
            "continue_final_message": continue_final_message,
        }

        template_kwargs.update(self.tokenizer_config)
        template_kwargs.update(kwargs)

        return template_kwargs

    @staticmethod
    def _unescape_rendered_template(text: str) -> str:
        return html.unescape(text)

    def _render_template(
        self,
        messages,
        tools,
        documents,
        add_generation_prompt,
        continue_final_message,
        **kwargs,
    ):
        template_kwargs = self._build_template_kwargs(
            messages=messages,
            tools=tools,
            documents=documents,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            **kwargs,
        )

        rendered = self.template.render(**template_kwargs)

        return self._unescape_rendered_template(rendered)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        documents=None,
        add_generation_prompt=False,
        continue_final_message=False,
        tokenize=False,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        return_dict=False,
        **kwargs,
    ):
        if add_generation_prompt and continue_final_message:
            raise ValueError(
                "add_generation_prompt=True and "
                "continue_final_message=True are mutually exclusive."
            )

        if return_tensors is not None:
            raise ValueError("return_tensors requires tokenize=True.")

        if return_dict:
            raise ValueError("return_dict requires tokenize=True.")

        if tokenize:
            raise NotImplementedError("Tokenization is not implemented.")

        return self._render_template(
            messages=conversation,
            tools=tools,
            documents=documents,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            **kwargs,
        )

    def parse_tags(self, text: str):
        pattern = re.compile(r"<([a-zA-Z_][\w-]*)>(.*?)</\1>", re.DOTALL)

        results = []
        last_end = 0

        for match in pattern.finditer(text):
            outside_text = text[last_end : match.start()]

            if outside_text.strip():
                results.append({"tag": None, "text": outside_text})

            results.append({"tag": match.group(1), "text": match.group(2)})

            last_end = match.end()

        outside_text = text[last_end:]

        if outside_text.strip():
            results.append({"tag": None, "text": outside_text})

        return results


# In[3]:


if __name__ == "__main__":
    import os

    folder_path = os.path.join(
        os.path.expanduser("~"), ".cache", config.DEFAULT_FOLDER, "LFM2.5-VL-450M-ONNX"
    )
    template = ChatTemplate(folder_path)

    messages = [
        {
            "role": "system",
            "content": "You are helpful. You have access to various tools",
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

    prompt = template.apply_chat_template(
        messages,
        tools,
        add_generation_prompt=True,
    )

    print(prompt)


# In[ ]:




