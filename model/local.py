"""Local Hugging Face Transformers inference for Qwen3 and its native template."""

from __future__ import annotations

from typing import Any, Sequence

from model.client import ChatMessage, ModelClient


class LocalTransformersClient(ModelClient):
    """Load a local Qwen-compatible causal LM lazily and generate raw text.

    ``local_files_only`` defaults to True so construction/inference never starts
    an implicit model download. Install the optional ``torch`` and
    ``transformers`` dependencies and make the checkpoint available locally
    before calling ``generate``. Qwen3's tokenizer chat template receives the
    standard tool schemas and emits its native ``<tool_call>`` representation.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-1.7B",
        *,
        device: str = "auto",
        dtype: str = "float16",
        max_new_tokens: int = 512,
        enable_thinking: bool = False,
        local_files_only: bool = True,
    ) -> None:
        if not model_path:
            raise ValueError("model_path must not be empty")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if dtype not in {"float16", "bfloat16", "float32", "auto"}:
            raise ValueError("dtype must be float16, bfloat16, float32, or auto")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.local_files_only = local_files_only
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._resolved_device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LocalTransformersClient requires torch and transformers; "
                "no packages or model files were installed during Phase A2."
            ) from exc

        resolved_device = self.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        dtype_by_name = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype_by_name[self.dtype],
            "local_files_only": self.local_files_only,
            "trust_remote_code": False,
        }
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        model.to(resolved_device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = resolved_device

    def generate(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]],
        *,
        timeout: float | None = None,
    ) -> str:
        if timeout is not None and timeout <= 0:
            raise TimeoutError("no inference time remains")
        self._load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._resolved_device is not None

        rendered = self._tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            return_tensors="pt",
        )
        rendered = rendered.to(self._resolved_device)
        input_length = rendered.shape[-1]
        with self._torch.inference_mode():
            generation_options: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "use_cache": True,
            }
            if timeout is not None:
                # Transformers checks max_time between decoding steps.
                generation_options["max_time"] = timeout
            generated = self._model.generate(rendered, **generation_options)
        new_tokens = generated[0, input_length:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=False)
