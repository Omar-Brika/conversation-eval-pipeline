from typing import Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class LiveBrainState(TypedDict):
    session_id: str
    intent: str
    target_language: str
    recent_messages: Sequence[BaseMessage]
    text_command: str
    dut_audio: bytes
    dut_text: str
    routing_flag: Literal["AWAITING_DUT", "PROCEED_TO_JUDGE"]
    final_verdict: str
    final_reasoning: str
    turn_count: int
    llm_logs: list[dict]
