"""Local Hugging Face Transformers inference for Qwen3 and its native template."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
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


class LocalOpenAICompatibleClient(ModelClient):
    """Call a loopback OpenAI-compatible server without credentials or API keys.

    The endpoint is restricted to localhost so selecting this option cannot
    accidentally send prompts or tool schemas to a remote service. Structured
    function calls are converted to Qwen3's native ``<tool_call>`` text for the
    existing parser; plain assistant content is returned unchanged.
    """

    def __init__(self, endpoint: str, *, model: str = "Qwen3-1.7B") -> None:
        if not endpoint or not model:
            raise ValueError("endpoint and model must not be empty")
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError("model endpoint must be an HTTP(S) loopback URL without embedded credentials")
        self.endpoint = endpoint
        self.model = model

    def generate(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]],
        *,
        timeout: float | None = None,
    ) -> str:
        if timeout is not None and timeout <= 0:
            raise TimeoutError("no inference time remains")
        body = json.dumps(
            {
                "model": self.model,
                "messages": list(messages),
                "tools": list(tools),
                "tool_choice": "auto",
                "stream": False,
            }
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(f"local model endpoint returned HTTP {exc.code}: {detail}") from exc
        except TimeoutError:
            raise
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError("local model endpoint request timed out") from exc
            raise RuntimeError(f"local model endpoint request failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"local model endpoint request failed: {exc}") from exc

        try:
            message = payload["choices"][0]["message"]
        except (TypeError, KeyError, IndexError) as exc:
            raise RuntimeError("local model endpoint returned an invalid chat-completions response") from exc
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise RuntimeError("local model endpoint returned non-text assistant content")
        rendered_calls: list[str] = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("local model endpoint returned malformed tool-call arguments") from exc
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise RuntimeError("local model endpoint returned an invalid structured tool call")
            rendered_calls.append(
                "<tool_call>"
                + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, separators=(",", ":"))
                + "</tool_call>"
            )
        return (content or "") + "".join(rendered_calls)
