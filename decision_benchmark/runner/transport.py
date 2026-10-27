"""OpenRouter boundary: no redirects, hidden proxies or execution deadline."""

import base64
from datetime import datetime, timezone
import time
import urllib.error
import urllib.request
from ..encoding import encode, sha256

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class OpenRouterTransport:
    def __init__(self, api_key, allow_env_proxy=False):
        if not api_key or "\n" in api_key or "\r" in api_key:
            raise ValueError("Missing or invalid OPENROUTER_API_KEY")
        self.api_key = api_key
        self.allow_env_proxy = allow_env_proxy

    def send(self, body):
        handlers = [NoRedirect()]
        if not self.allow_env_proxy:
            handlers.append(urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers)
        request = urllib.request.Request(
            ENDPOINT,
            data=encode(body),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with opener.open(request, timeout=None) as response:
                status, data = response.status, response.read()
        except urllib.error.HTTPError as error:
            status = error.code
            try:
                data = error.read()
            except Exception:
                return self._unknown(started, "UnreadableHTTPBody")
        except Exception as error:
            return self._unknown(started, type(error).__name__)
        original_sha = sha256(data)
        safe_data = data.replace(self.api_key.encode(), b"<REDACTED>")
        result = {
            "status": status,
            "elapsed_ms": (time.monotonic() - started) * 1000,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "response_text": safe_data.decode("utf-8", "replace"),
            "body_sha256": sha256(safe_data),
            "body_encoding": "base64",
            "response_base64": base64.b64encode(safe_data).decode(),
            "unknown": status in {408, 504, 520, 522, 524},
        }
        if safe_data != data:
            result.update(redacted=True, original_body_sha256=original_sha)
        return result

    @staticmethod
    def _unknown(started, error_type):
        return {
            "status": None,
            "response_text": "",
            "unknown": True,
            "error": error_type,
            "elapsed_ms": (time.monotonic() - started) * 1000,
            "body_encoding": "utf8",
            "body_sha256": sha256(b""),
        }
