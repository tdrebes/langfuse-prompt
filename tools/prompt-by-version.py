from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from langfuse import Langfuse


class LangfusePromptByVersionTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        lf = Langfuse(
            host=self.runtime.credentials["langfuse_host"],
            secret_key=self.runtime.credentials["langfuse_secret_key"],
            public_key=self.runtime.credentials["langfuse_public_key"],
        )

        version = tool_parameters["prompt_version"] if tool_parameters["prompt_version"] > 0 else None
        label = tool_parameters["prompt_label"] if tool_parameters["prompt_label"] else None
        prompt = lf.get_prompt(
            name=tool_parameters["prompt_name"],
            version=version,
            label=label,
        )
        if prompt is None:
            raise ValueError(f"Prompt {tool_parameters['prompt_name']} not found")

        # TODO: cache & support compiled prompts (maybe separate tool) & langfuse observability

        yield self.create_text_message(prompt.prompt)
