from typing import Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class BrainState(TypedDict):
    session_id: str
    intent: str
    target_language: str
    recent_messages: Sequence[BaseMessage]
    dut_script: list[str]
    dut_script_idx: int
    routing_flag: Literal["AWAITING_DUT", "PROCEED_TO_JUDGE", "TERMINATED"]
    final_verdict: str
    final_reasoning: str
    # Metrics
    turn_count: int
    expected_turns: int
    llm_logs: list[dict]
