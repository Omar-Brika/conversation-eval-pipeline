import logging
import uuid
from typing import Callable

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.errors import GraphRecursionError
from langgraph.checkpoint.memory import InMemorySaver
from live_nodes import (
    create_asr_node,
    create_judge_node,
    create_prompter_node,
    create_voice_cmd_node,
)
from live_state import LiveBrainState
from llm_tracker import LLMTimeTracker


logger = logging.getLogger(__name__)


def _ensure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)-5s | %(message)s",
        )


class LiveConversationGraph:
    def __init__(
        self,
        llm: ChatOpenAI,
        voice_command_fn: Callable[[str], bytes],
        speech_recognition_fn: Callable[[bytes], str],
    ):
        _ensure_logging()
        self.llm = llm
        self.tracker = self._ensure_tracker(llm)
        self.checkpointer = InMemorySaver()
        self._graph = self._build(voice_command_fn, speech_recognition_fn)

    @staticmethod
    def _ensure_tracker(llm: ChatOpenAI) -> LLMTimeTracker:
        callbacks = llm.callbacks
        if callbacks is None:
            tracker = LLMTimeTracker()
            llm.callbacks = [tracker]
            return tracker

        from langchain_core.callbacks import BaseCallbackManager

        if isinstance(callbacks, BaseCallbackManager):
            for h in callbacks.handlers:
                if isinstance(h, LLMTimeTracker):
                    return h
            tracker = LLMTimeTracker()
            callbacks.add_handler(tracker)
            return tracker

        for h in callbacks:
            if isinstance(h, LLMTimeTracker):
                return h
        tracker = LLMTimeTracker()
        callbacks.append(tracker)
        return tracker

    def _build(
        self,
        voice_command_fn: Callable[[str], bytes],
        speech_recognition_fn: Callable[[bytes], str],
    ) -> CompiledStateGraph:
        prompter = create_prompter_node(self.llm)
        voice_cmd = create_voice_cmd_node(voice_command_fn)
        asr = create_asr_node(speech_recognition_fn)
        judge = create_judge_node(self.llm)

        workflow = StateGraph(LiveBrainState)
        workflow.add_node("prompter", prompter)
        workflow.add_node("voice_cmd", voice_cmd)
        workflow.add_node("asr", asr)
        workflow.add_node("judge", judge)

        workflow.add_edge(START, "prompter")
        workflow.add_conditional_edges(
            "prompter",
            lambda state: state["routing_flag"],
            {
                "AWAITING_DUT": "voice_cmd",
                "PROCEED_TO_JUDGE": "judge",
            },
        )
        workflow.add_edge("voice_cmd", "asr")
        workflow.add_edge("asr", "prompter")
        workflow.add_edge("judge", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def invoke(self, intent: str, language: str, max_turns: int = 10) -> dict:
        session_id = str(uuid.uuid4())
        config = {
                "configurable": {"thread_id": session_id},
                "recursion_limit": 3 * max_turns,
            }
        initial_state: LiveBrainState = {
            "session_id": session_id,
            "intent": intent,
            "target_language": language,
            "recent_messages": [],
            "text_command": "",
            "dut_audio": b"",
            "dut_text": "",
            "routing_flag": "AWAITING_DUT",
            "final_verdict": "ERROR",
            "final_reasoning": "",
            "turn_count": 0,
            "llm_logs": [],
        }
        try:
            final_state = self._graph.invoke(initial_state, config)
            return {
                "final_verdict": final_state["final_verdict"],
                "final_reasoning": final_state["final_reasoning"],
                "turn_count": final_state["turn_count"],
                "llm_logs": final_state["llm_logs"],
            }
        except GraphRecursionError:
            last_good = self._graph.get_state(config).values
            turn_count = last_good.get("turn_count", 0)
            logger.warning(
                            f"hit max_turns={max_turns} "
                            f"(turn_count={turn_count}) without the DUT completing."
            )
            return {
                           "final_verdict": "FAIL",
                           "final_reasoning": (
                               f"Exceeded max_turns ({max_turns}) without the DUT "
                               f"completing the request or explicitly refusing it."
                           ),
                           "turn_count": turn_count,
                           "llm_logs": last_good.get("llm_logs", []),
                       }

    def is_passed(self, intent: str, language: str, max_turns: int = 10) -> bool:
        return self.invoke(intent, language, max_turns)["final_verdict"] == "PASS"
