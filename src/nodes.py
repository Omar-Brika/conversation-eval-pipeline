from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from llm_tracker import LLMTimeTracker
from parsers import parse_judge_response, parse_prompter_response
from state import BrainState


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


def create_prompter_node(llm: ChatOpenAI, verbose: bool):
    def prompter_node(state: BrainState):
        idx = state.get("dut_script_idx", 0)
        dut_script = state.get("dut_script", [])
        recent_msgs = state.get("recent_messages", [])
        current_logs = state.get("llm_logs", [])

        turn_count = state.get("turn_count", 0) + 1

        if not recent_msgs:
            sys_prompt = SystemMessage(
                content=(
                    f"You are a Test User. Language: {state['target_language']}. "
                    f"Intent: {state['intent']}. Generate ONLY the natural voice command. "
                    f'Output JSON: {{"text_command": "...", "is_interaction_complete": false}}'
                )
            )

            response, llm_time = _invoke_with_tracking(
                llm,
                [sys_prompt, HumanMessage(content="Generate.")],
                ["initial_prompter"],
            )
            metric_type = "initial_prompter"

            raw_content = response.content
            response_text = (
                raw_content if isinstance(raw_content, str) else str(raw_content)
            )

            parsed = parse_prompter_response(response_text, state["intent"])
            user_cmd = parsed.text_command
            is_complete = False
            dut_response_text = "N/A (First Turn)"
        else:
            last_dut_msg = recent_msgs[-1].content
            dut_response_text = last_dut_msg

            conversation_history = _format_conversation(recent_msgs)
            sys_prompt = SystemMessage(
                content=(
                    f"You are a Test User. Intent: {state['intent']}. Language: {state['target_language']}.\n"
                    f"Conversation History:\n{conversation_history}\n\n"
                    f'The Assistant just said: "{last_dut_msg}"\n'
                    f"Did the assistant complete the task or refuse? If yes, set is_interaction_complete to true.\n"
                    f"If the assistant asked for clarification, set it to false and generate the next command.\n"
                    f'Output JSON: {{"text_command": "...", "is_interaction_complete": true/false}}'
                )
            )

            response, llm_time = _invoke_with_tracking(
                llm,
                [sys_prompt, HumanMessage(content="Evaluate and reply.")],
                ["routing_voice_cmd"],
            )
            metric_type = "routing_voice_cmd"

            raw_content = response.content
            response_text = (
                raw_content if isinstance(raw_content, str) else str(raw_content)
            )

            parsed = parse_prompter_response(response_text, state["intent"])
            user_cmd = parsed.text_command
            is_complete = parsed.is_interaction_complete

        log_entry = {
            "node": "prompter",
            "turn": turn_count,
            "llm_time_sec": round(llm_time, 4),
            "metric_type": metric_type,
            "dut_response": dut_response_text,
            "raw_llm_output": response.content,
            "parsed_decision": {
                "text_command": parsed.text_command,
                "is_interaction_complete": parsed.is_interaction_complete,
            },
        }
        current_logs.append(log_entry)

        if verbose:
            print(f"    [DEBUG] Turn {turn_count} | LLM Time: {llm_time:.3f}s")
            if idx > 0:
                print(
                    f'           |- DUT Response: "{dut_response_text[:70]}{"..." if len(dut_response_text) > 70 else ""}"'
                )
            print(
                f'           |- Router Decision: is_complete={is_complete} | Next Cmd: "{user_cmd[:60]}{"..." if len(user_cmd) > 60 else ""}"'
            )

        if idx < len(dut_script):
            dut_response = dut_script[idx]
            new_idx = idx + 1
        else:
            dut_response = "I didn't understand."
            new_idx = idx
            is_complete = True

        tester_msg = HumanMessage(content=user_cmd, name="Tester")
        dut_msg = AIMessage(content=dut_response, name="DUT")

        current_msgs = list(recent_msgs)
        recent_msgs_updated = current_msgs + [tester_msg, dut_msg]

        if is_complete or new_idx >= len(dut_script):
            next_flag = "PROCEED_TO_JUDGE"
        else:
            next_flag = "AWAITING_DUT"

        return {
            "recent_messages": recent_msgs_updated,
            "dut_script_idx": new_idx,
            "turn_count": turn_count,
            "routing_flag": next_flag,
            "llm_logs": current_logs,
        }

    return prompter_node


def create_judge_node(llm: ChatOpenAI, verbose: bool):
    def judge_node(state: BrainState):
        msgs = state.get("recent_messages", [])
        conversation_log = _format_conversation(msgs)
        current_logs = state.get("llm_logs", [])

        prompt = f"""You are a QA Judge.
Intent: {state["intent"]}
Language: {state["target_language"]}

Conversation History:
{conversation_log}

Task: Did the DUT successfully fulfill the user's intent?
Output ONLY JSON: {{"verdict": "PASS" or "FAIL", "reasoning": "short explanation"}}
"""

        response, llm_time = _invoke_with_tracking(
            llm,
            [HumanMessage(content=prompt)],
            ["judgment"],
        )
        metric_type = "judgment"

        raw_content = response.content
        response_text = raw_content if isinstance(raw_content, str) else str(raw_content)
        parsed = parse_judge_response(response_text)

        log_entry = {
            "node": "judge",
            "llm_time_sec": round(llm_time, 4),
            "metric_type": metric_type,
            "raw_llm_output": response.content,
            "parsed_decision": {"verdict": parsed.verdict, "reasoning": parsed.reasoning},
        }
        current_logs.append(log_entry)

        if verbose:
            print(f"    [DEBUG] Judge Evaluation | LLM Time: {llm_time:.3f}s")
            print(
                f'           |- Verdict: {parsed.verdict} | Reasoning: "{parsed.reasoning[:70]}{"..." if len(parsed.reasoning) > 70 else ""}"'
            )

        return {
            "routing_flag": "TERMINATED",
            "final_verdict": parsed.verdict,
            "final_reasoning": parsed.reasoning,
            "llm_logs": current_logs,
        }

    return judge_node
