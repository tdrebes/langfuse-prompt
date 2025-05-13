import json
from datetime import datetime, timedelta, timezone
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from langfuse import Langfuse

class LangfusePromptByLabelTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        prompt_name = tool_parameters.get("prompt_name")
        prompt_label = tool_parameters.get("prompt_label")
        ttl = tool_parameters.get("cache_ttl", 0)

        if ttl == 0:
            prompt_text = self._fetch_and_store(
                prompt_name, prompt_label,
                f"prompt_{prompt_name}:{prompt_label}"
            )
            yield self.create_text_message(prompt_text)
            return

        cache_key = f"prompt_{prompt_name}:{prompt_label}"
        raw = None
        try:
            raw = self.session.storage.get(cache_key)
        except Exception:
            raw = None

        if raw:
            try:
                payload = json.loads(raw.decode("utf-8"))
                saved_at = datetime.fromisoformat(payload["timestamp"])
                if datetime.now(timezone.utc) - saved_at < timedelta(minutes=ttl):
                    prompt_text = payload["prompt"]
                else:
                    raise ValueError
            except Exception:
                prompt_text = self._fetch_and_store(prompt_name, prompt_label, cache_key)
        else:
            prompt_text = self._fetch_and_store(prompt_name, prompt_label, cache_key)

        yield self.create_text_message(prompt_text)

    def _fetch_and_store(self, name: str, label: str, cache_key: str) -> str:
        lf = Langfuse(
            host=self.runtime.credentials["langfuse_host"],
            secret_key=self.runtime.credentials["langfuse_secret_key"],
            public_key=self.runtime.credentials["langfuse_public_key"],
        )
        prompt_obj = lf.get_prompt(name=name, label=label)
        payload = {
            "prompt": prompt_obj.prompt,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        try:
            self.session.storage.set(cache_key, json.dumps(payload).encode("utf-8"))
        except Exception:
            pass
        return prompt_obj.prompt
