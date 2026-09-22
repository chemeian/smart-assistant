"""
Unified LLM Service - supports DeepSeek & Tongyi Qianwen via OpenAI-compatible API.
Includes multimodal (image) support and single/multi-turn conversation.
"""
import json
import time
from typing import List, Dict, Optional

import requests
from config import config


class LLMService:
    """Multi-provider LLM service with fallback and multimodal support."""

    PROVIDERS = {
        "deepseek": {
            "api_base": config.DEEPSEEK_API_BASE,
            "api_key": config.DEEPSEEK_API_KEY,
            "model": config.DEEPSEEK_MODEL,
            "supports_vision": False,
        },
        "qwen": {
            "api_base": config.QWEN_API_BASE,
            "api_key": config.QWEN_API_KEY,
            "model": config.QWEN_MODEL,
            "supports_vision": True,
        },
    }

    def __init__(self, provider: str = "deepseek"):
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. "
                             f"Choose from: {list(self.PROVIDERS.keys())}")
        self._provider_name = provider
        self._cfg = self.PROVIDERS[provider]

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def supports_vision(self) -> bool:
        return self._cfg.get("supports_vision", False)

    # ----------------------------------------------------------------
    def chat(self, messages: List[Dict], stream: bool = False,
             temperature: float = 0.7, max_tokens: int = 2048) -> Dict:
        """Send a chat completion request and return the response dict."""
        if not self._cfg["api_key"]:
            return {
                "role": "assistant",
                "content": f"[{self._cfg['model']}] API key not configured. "
                           f"Set {self._provider_name.upper()}_API_KEY in .env",
                "provider": self._provider_name,
            }

        url = f"{self._cfg['api_base'].rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._cfg['api_key']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._cfg["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        try:
            resp = requests.post(
                url, headers=headers, json=payload,
                timeout=config.CHAT_TIMEOUT, stream=stream
            )
            resp.raise_for_status()

            if stream:
                return self._handle_stream(resp)

            data = resp.json()
            choice = data["choices"][0]
            return {
                "role": choice["message"]["role"],
                "content": choice["message"]["content"],
                "provider": self._provider_name,
                "model": data.get("model", self._cfg["model"]),
                "usage": data.get("usage", {}),
            }

        except requests.exceptions.Timeout:
            return {"role": "assistant",
                    "content": "Request timed out. Please try again later.",
                    "provider": self._provider_name}
        except requests.exceptions.RequestException as e:
            return {"role": "assistant",
                    "content": f"API request failed: {str(e)}",
                    "provider": self._provider_name}
        except (KeyError, json.JSONDecodeError) as e:
            return {"role": "assistant",
                    "content": f"Response parsing failed: {str(e)}",
                    "provider": self._provider_name}

    # ----------------------------------------------------------------
    def chat_with_image(self, messages: List[Dict], image_data_uri: str,
                        image_detail: str = "auto",
                        temperature: float = 0.7, max_tokens: int = 2048) -> Dict:
        """
        Send a multimodal chat request with an image attachment.
        Uses OpenAI-compatible vision format for providers that support it.
        Falls back to text-only with a note for providers without vision.
        """
        if not self._cfg["api_key"]:
            return {
                "role": "assistant",
                "content": f"[{self._cfg['model']}] API key not configured.",
                "provider": self._provider_name,
            }

        if not self._cfg.get("supports_vision", False):
            # Provider does not support vision - fall back gracefully
            note = ("[Note: {provider} does not support image input. "
                    "Please use text only.]").format(provider=self._provider_name)
            if messages:
                messages[-1]["content"] = note + "\n" + str(messages[-1].get("content", ""))
            return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

        url = f"{self._cfg['api_base'].rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._cfg['api_key']}",
            "Content-Type": "application/json",
        }

        # Build multimodal message for Qwen / vision-capable models
        last_msg = messages[-1] if messages else {"role": "user", "content": "Describe this image."}
        text_content = last_msg.get("content", "Describe this image.")
        if isinstance(text_content, list):
            text_content = text_content[0].get("text", "")

        last_msg["content"] = [
            {"type": "text", "text": text_content},
            {"type": "image_url", "image_url": {"url": image_data_uri, "detail": image_detail}},
        ]
        messages[-1] = last_msg

        payload = {
            "model": self._cfg["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            resp = requests.post(
                url, headers=headers, json=payload,
                timeout=max(config.CHAT_TIMEOUT, 120)
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            return {
                "role": choice["message"]["role"],
                "content": choice["message"]["content"],
                "provider": self._provider_name,
                "model": data.get("model", self._cfg["model"]),
                "usage": data.get("usage", {}),
            }
        except requests.exceptions.Timeout:
            return {"role": "assistant",
                    "content": "Image analysis request timed out.",
                    "provider": self._provider_name}
        except requests.exceptions.RequestException as e:
            return {"role": "assistant",
                    "content": f"Image request failed: {str(e)}",
                    "provider": self._provider_name}

    # ----------------------------------------------------------------
    def _handle_stream(self, resp) -> Dict:
        """Read SSE stream and concatenate content chunks."""
        content_parts = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or line.startswith(":"):
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta:
                        content_parts.append(delta["content"])
                except json.JSONDecodeError:
                    continue
        return {
            "role": "assistant",
            "content": "".join(content_parts),
            "provider": self._provider_name,
        }

    # ----------------------------------------------------------------
    @staticmethod
    def list_models() -> Dict:
        """Return available models and their status."""
        providers = LLMService.PROVIDERS
        return {
            name: {
                "model": info["model"],
                "configured": bool(info["api_key"]),
                "supports_vision": info.get("supports_vision", False),
            }
            for name, info in providers.items()
        }

    # ----------------------------------------------------------------
    @staticmethod
    def switch_provider(provider: str) -> "LLMService":
        """Factory convenience: return a new service instance."""
        return LLMService(provider)