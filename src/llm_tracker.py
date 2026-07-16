import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


class LLMTimeTracker(BaseCallbackHandler):
    """Catches every LLM call (including retries) and measures exact execution time."""

    def __init__(self):
        self.calls = []
        self._start_times: Dict[UUID, float] = {}
        self._run_tags: Dict[UUID, List[str]] = {}

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running."""
        self._start_times[run_id] = time.time()
        self._run_tags[run_id] = tags or []

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when standard LLM starts running."""
        if run_id not in self._start_times:
            self._start_times[run_id] = time.time()
            self._run_tags[run_id] = tags or []

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running."""
        start_time = self._start_times.pop(run_id, None)
        tags = self._run_tags.pop(run_id, [])

        if start_time is not None:
            duration = time.time() - start_time
            tokens = {}
            if response.llm_output:
                tokens = response.llm_output.get("token_usage", {})

            self.calls.append(
                {
                    "duration": duration,
                    "tokens": tokens,
                    "tags": tags,
                    "run_id": str(run_id),
                }
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors."""
        self._start_times.pop(run_id, None)
        self._run_tags.pop(run_id, None)
