from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError


class LangfusePromptProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            host = credentials.get("langfuse_host")
            secret_key = credentials.get("langfuse_secret_key")
            
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
