#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import tempfile
from abc import ABC, abstractmethod
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from .utils import hf_download


# In[4]:


class ONNXBase(ABC):
    """Base class for models running with ONNX Runtime."""

    _cuda_works: bool | None = None

    def __init__(
        self,
        repo_id: str,
        precision: str = "fp16",
        required_files=None,
        save_dir=None,
    ):
        self.repo_id = repo_id
        self.precision = precision

        self._save_dir = hf_download(
            repo_id=repo_id,
            precision=precision,
            required_files=required_files,
            save_dir=save_dir,
        )

    # ------------------------------------------------------------------
    # Model-specific API
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_config(self):
        raise NotImplementedError

    @abstractmethod
    def _load_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def _get_onnx_path(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def initialize_cache(self):
        raise NotImplementedError

    @abstractmethod
    def update_cache(self, cache, outputs):
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
    ):
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
    ):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # ONNX Runtime infrastructure
    # ------------------------------------------------------------------

    @classmethod
    def _check_cuda(cls) -> bool:
        """Check whether CUDAExecutionProvider is functional."""

        if cls._cuda_works is not None:
            return cls._cuda_works

        available = ort.get_available_providers()

        if "CUDAExecutionProvider" not in available:
            cls._cuda_works = False
            return False

        try:
            X = helper.make_tensor_value_info(
                "X",
                TensorProto.FLOAT,
                [1],
            )

            Y = helper.make_tensor_value_info(
                "Y",
                TensorProto.FLOAT,
                [1],
            )

            node = helper.make_node(
                "Identity",
                ["X"],
                ["Y"],
            )

            graph = helper.make_graph(
                [node],
                "cuda_test",
                [X],
                [Y],
            )

            model = helper.make_model(
                graph,
                opset_imports=[
                    helper.make_opsetid("", 13),
                ],
            )

            model.ir_version = 8

            with tempfile.NamedTemporaryFile(
                suffix=".onnx",
                delete=True,
            ) as f:
                onnx.save(model, f.name)

                ort.InferenceSession(
                    f.name,
                    providers=["CUDAExecutionProvider"],
                )

            cls._cuda_works = True

        except Exception:
            cls._cuda_works = False

        return cls._cuda_works

    def get_providers(self) -> list[str]:
        """Get ONNX Runtime execution providers."""

        cls = type(self)

        if cls._check_cuda():
            return [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

        return [
            "CPUExecutionProvider",
        ]

    def load_onnx_session(
        self,
        path: str,
        providers: list[str] | None = None,
    ):
        """Create an ONNX Runtime inference session."""

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        opts = ort.SessionOptions()

        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        opts.enable_mem_pattern = False
        opts.enable_cpu_mem_arena = False

        if providers is None:
            providers = self.get_providers()

        try:
            return ort.InferenceSession(
                path,
                sess_options=opts,
                providers=providers,
            )

        except Exception:
            if (
                "CUDAExecutionProvider" in providers
                and "CPUExecutionProvider" in providers
            ):
                return ort.InferenceSession(
                    path,
                    sess_options=opts,
                    providers=[
                        "CPUExecutionProvider",
                    ],
                )

            raise

    def load_session(self):
        """Load the model's ONNX session."""

        path = self._get_onnx_path()

        self.session = self.load_onnx_session(path)

        self.input_names = {inp.name for inp in self.session.get_inputs()}

        self.output_infos = self.session.get_outputs()

        return self.session


# In[ ]:




