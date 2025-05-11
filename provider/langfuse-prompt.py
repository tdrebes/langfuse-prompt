from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from langfuse import Langfuse

class LangfusePromptProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            host = credentials.get("langfuse_host")
            secret_key = credentials.get("langfuse_secret_key")
            public_key = credentials.get("langfuse_public_key")
        
            lf = Langfuse(host=host, secret_key=secret_key, public_key=public_key)
            lf.auth_check()
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
