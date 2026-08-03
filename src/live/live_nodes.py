import logging
from typing import Callable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from live_state import LiveBrainState
from llm_tracker import LLMTimeTracker
from parsers import parse_judge_response, parse_prompter_response

logger = logging.getLogger(__name__)


def _get_tracker_from_llm(llm: ChatOpenAI) -> LLMTimeTracker:
    for callback in llm.callbacks:
        if isinstance(callback, LLMTimeTracker):
            return callback
    raise ValueError("No LLMTimeTracker found in LLM callbacks")


def _invoke_with_tracking(
    llm: ChatOpenAI,
    messages: Sequence[BaseMessage],
    tags: list[str],
) -> tuple[BaseMessage, float]:
    tracker = _get_tracker_from_llm(llm)
    start_idx = len(tracker.calls)
    response = llm.invoke(messages, config={"tags": tags})
    end_idx = len(tracker.calls)
    llm_time = sum(c["duration"] for c in tracker.calls[start_idx:end_idx])
    return response, llm_time


def _format_conversation(messages: Sequence[BaseMessage]) -> str:
    parts = []
    for m in messages:
        name = m.name if m.name is not None else "Unknown"
        parts.append(f"{name}: {m.content}")
    return "\n".join(parts)


def create_prompter_node(llm: ChatOpenAI):
    def prompter_node(state: LiveBrainState):
        recent_msgs = list(state.get("recent_messages", []))
        intent = state["intent"]
        language = state["target_language"]
        current_logs = list(state.get("llm_logs", []))
        turn_count = state.get("turn_count", 0) + 1

        is_first_turn = len(recent_msgs) == 0

        if is_first_turn:
            conversation_history = "<EMPTY>"
            dut_response_text = "N/A (First Turn)"
            metric_type = "initial_prompter"
            tags = ["initial_prompter"]
        else:
            dut_text = state.get("dut_text", "")
            recent_msgs = recent_msgs + [AIMessage(content=dut_text, name="DUT")]
            conversation_history = _format_conversation(recent_msgs)
            dut_response_text = dut_text
            metric_type = "routing_voice_cmd"
            tags = ["routing_voice_cmd"]

        sys_prompt = SystemMessage(
            content=(
                f"You are a Test User simulating a human driver.\n"
                f"Language: {language}\n"
                f"Intent: {intent}\n\n"
                f"If this is the first turn, generate the very first natural voice "
                f"command the user would say.\n\n"
                f"Otherwise:\n"
                f"- If the assistant actually performed or explicitly confirmed the "
                f"requested action, or explicitly refused / said it cannot help, "
                f'output an empty text_command: "".\n'
                f"- If the assistant asked a clarifying question, requested a choice, "
                f"asked for confirmation, or requested missing information, generate "
                f"the next natural voice command that answers the assistant so the "
                f"task can continue.\n"
                f"- Merely listing options is NOT task completion.\n\n"
                f'Output ONLY JSON: {{"text_command": "..."}}'
            )
        )

        human_prompt = HumanMessage(
            content=(
                f"First Turn: {is_first_turn}\n\n"
                f"Conversation History:\n"
                f"{conversation_history}\n\n"
                f"Latest Assistant Response:\n"
                f"{dut_response_text}"
            )
        )

        response, llm_time = _invoke_with_tracking(
            llm,
            [sys_prompt, human_prompt],
            tags,
        )

        raw_content = response.content
        response_text = (
            raw_content if isinstance(raw_content, str) else str(raw_content)
        )

        parsed = parse_prompter_response(response_text, intent)
        text_command = (parsed.text_command or "").strip()

        is_complete = (not is_first_turn) and text_command == ""

        if is_complete:
            updated_messages = recent_msgs
        else:
            tester_msg = HumanMessage(content=text_command, name="Tester")
            updated_messages = recent_msgs + [tester_msg]

        routing_flag = (
            "PROCEED_TO_JUDGE" if is_complete else "AWAITING_DUT"
        )

        log_entry = {
            "node": "prompter",
            "turn": turn_count,
            "llm_time_sec": round(llm_time, 4),
            "metric_type": metric_type,
            "dut_response": dut_response_text,
            "raw_llm_output": response.content,
            "parsed_decision": {
                "text_command": text_command,
                "is_interaction_complete": is_complete,
            },
        }
        current_logs.append(log_entry)

        logger.info(
            f"Prompter Turn {turn_count} | LLM Time: {llm_time:.3f}s | "
            f'DUT: "{dut_response_text}" | '
            f'Next Cmd: "{text_command}" | Complete: {is_complete}'
        )

        return {
            "recent_messages": updated_messages,
            "text_command": text_command,
            "routing_flag": routing_flag,
            "turn_count": turn_count,
            "llm_logs": current_logs,
        }

    return prompter_node


def create_voice_cmd_node(voice_command_fn: Callable[[str], bytes]):
    def voice_cmd_node(state: LiveBrainState):
        text_command = state["text_command"]
        audio = voice_command_fn(text_command)
        return {"dut_audio": audio}

    return voice_cmd_node


def create_asr_node(speech_recognition_fn: Callable[[bytes], str]):
    def asr_node(state: LiveBrainState):
        audio = state["dut_audio"]
        text = speech_recognition_fn(audio)
        return {"dut_text": text}

    return asr_node


def create_judge_node(llm: ChatOpenAI):
    def judge_node(state: LiveBrainState):
        msgs = list(state.get("recent_messages", []))
        conversation = _format_conversation(msgs)
        intent = state["intent"]
        current_logs = list(state.get("llm_logs", []))

        prompt = (
            f"You are a QA Judge.\n"
            f"Intent: {intent}\n"
            f"Language: {state['target_language']}\n\n"
            f"Conversation History:\n{conversation}\n\n"
            f"Task: Did the DUT successfully fulfill the user's intent?\n"
            f'Output ONLY JSON: {{"verdict": "PASS" or "FAIL", "reasoning": "short explanation"}}'
        )

        response, llm_time = _invoke_with_tracking(
            llm, [HumanMessage(content=prompt)], ["judgment"]
        )

        raw_content = response.content
        response_text = raw_content if isinstance(raw_content, str) else str(raw_content)
        parsed = parse_judge_response(response_text)

        log_entry = {
            "node": "judge",
            "llm_time_sec": round(llm_time, 4),
            "metric_type": "judgment",
            "raw_llm_output": response.content,
            "parsed_decision": {"verdict": parsed.verdict, "reasoning": parsed.reasoning},
        }
        current_logs.append(log_entry)

        logger.info(
            f"Judge | LLM Time: {llm_time:.3f}s | "
            f'Verdict: {parsed.verdict} | Reasoning: "{parsed.reasoning[:70]}"'
        )

        return {
            "final_verdict": parsed.verdict,
            "final_reasoning": parsed.reasoning,
            "llm_logs": current_logs,
        }

    return judge_node
