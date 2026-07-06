import argparse
import json
import re
import sys
import time
from uuid import UUID
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Sequence, TypedDict, Any, Optional
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# ==============================================================================
# CONFIGURATION
# ==============================================================================


class Config:
    def __init__(self, model_name: str, port: int, test_samples: str, verbose: bool):
        self.model_name = model_name
        self.test_samples = test_samples
        self.verbose = verbose
        self.llm_config = {
            "base_url": f"http://localhost:{port}/v1",
            "api_key": "notneeded",
            "model": model_name,
            "temperature": 0.0,
            "max_retries": 2,
            "max_tokens": 512,
        }


TEST_SAMPLES = "test_cases.json"
VERBOSE = True

llm: ChatOpenAI | None = None
llm_tracker: LLMTimeTracker | None = None

# ==============================================================================
# LLM TRACKER (BaseCallHandler)
# ==============================================================================


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


# ==============================================================================
# 1. PYDANTIC SCHEMAS
# ==============================================================================
class PrompterSchema(BaseModel):
    text_command: str = Field(description="Exact voice command to send to the DUT.")
    is_interaction_complete: bool = Field(
        description="True if the DUT's response concludes the interaction (task done, or explicitly failed). False if DUT asked for clarification."
    )


class JudgeSchema(BaseModel):
    reasoning: str = Field(description="Brief evaluation of semantic equivalence.")
    verdict: Literal["PASS", "FAIL"]


# ==============================================================================
# 2. STATE DEFINITION
# ==============================================================================
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
    hallucination_count: int


# ==============================================================================
# 3. JSON PARSING
# ==============================================================================
def extract_json_from_text(text: str) -> dict | None:
    """JSON extraction"""
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass

    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass

    if '"text_command"' in text or '"verdict"' in text:
        try:
            cleaned = re.sub(r",\s*}", "}", text)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass

    return None


def parse_prompter_response(text: str, default_intent: str) -> PrompterSchema:
    json_data = extract_json_from_text(text)
    if json_data and "text_command" in json_data:
        return PrompterSchema(
            text_command=json_data["text_command"],
            is_interaction_complete=bool(
                json_data.get("is_interaction_complete", False)
            ),
        )

    clean_text = text.strip().replace('"', "").split("\n")[0]
    return PrompterSchema(
        text_command=clean_text or default_intent, is_interaction_complete=False
    )


def parse_judge_response(text: str) -> JudgeSchema:
    json_data = extract_json_from_text(text)
    if json_data and "verdict" in json_data:
        verdict_raw = str(json_data.get("verdict", "FAIL")).upper()
        verdict = "PASS" if "PASS" in verdict_raw else "FAIL"
        return JudgeSchema(
            verdict=verdict, reasoning=json_data.get("reasoning", "No reasoning")
        )

    text_lower = text.lower()
    verdict = "FAIL"
    if "pass" in text_lower and "fail" not in text_lower:
        verdict = "PASS"

    return JudgeSchema(verdict=verdict, reasoning=text.strip()[:150])


# ==============================================================================
# 4. NODE DEFINITIONS
# ==============================================================================
def prompter_node(state: BrainState):
    """
    Acts as the Decision Engine. Generates commands, simulates DUT,
    and decides whether to loop or proceed to Judge.
    """
    if llm is None:
        raise RuntimeError("LLM is not initialized.")
    idx = state.get("dut_script_idx", 0)
    dut_script = state.get("dut_script", [])
    recent_msgs = state.get("recent_messages", [])
    current_logs = state.get("llm_logs", [])
    hallucinations = state.get("hallucination_count", 0)

    # Increment turn count and get expected turns
    turn_count = state.get("turn_count", 0) + 1
    expected_turns = state.get("expected_turns", 0)

    # 1. Generate Command & Evaluate Routing
    if not recent_msgs:
        # Turn 1: generate initial command.
        sys_prompt = SystemMessage(
            content=(
                f"You are a Test User. Language: {state['target_language']}. "
                f"Intent: {state['intent']}. Generate ONLY the natural voice command. "
                f'Output JSON: {{"text_command": "...", "is_interaction_complete": false}}'
            )
        )
        assert llm_tracker is not None, "LLM Tracker not initialized"

        start_idx = len(llm_tracker.calls)
        response = llm.invoke(
            [sys_prompt, HumanMessage(content="Generate.")],
            config={"tags": ["initial_prompter"]},
        )
        end_idx = len(llm_tracker.calls)
        llm_time = sum(c["duration"] for c in llm_tracker.calls[start_idx:end_idx])
        metric_type = "initial_prompter"

        # extracting string frm response.content
        raw_content = response.content
        response_text = (
            raw_content if isinstance(raw_content, str) else str(raw_content)
        )

        parsed = parse_prompter_response(response_text, state["intent"])
        user_cmd = parsed.text_command
        is_complete = False
        dut_response_text = "N/A (First Turn)"
    else:
        # Turn 2+: Evaluate last DUT response and generate next command
        last_dut_msg = recent_msgs[-1].content
        dut_response_text = last_dut_msg

        sys_prompt = SystemMessage(
            content=(
                f"You are a Test User. Intent: {state['intent']}. Language: {state['target_language']}.\n"
                f'The Assistant just said: "{last_dut_msg}"\n'
                f"Did the assistant complete the task or refuse? If yes, set is_interaction_complete to true.\n"
                f"If the assistant asked for clarification, set it to false and generate the next command.\n"
                f'Output JSON: {{"text_command": "...", "is_interaction_complete": true/false}}'
            )
        )
        assert llm_tracker is not None, "LLM Tracker not initialized"
        start_idx = len(llm_tracker.calls)
        response = llm.invoke(
            [sys_prompt, HumanMessage(content="Evaluate and reply.")],
            config={"tags": ["routing_voice_cmd"]},
        )
        end_idx = len(llm_tracker.calls)
        llm_time = sum(c["duration"] for c in llm_tracker.calls[start_idx:end_idx])
        metric_type = "routing_voice_cmd"

        # extrating string from respose.content
        raw_content = response.content
        response_text = (
            raw_content if isinstance(raw_content, str) else str(raw_content)
        )

        parsed = parse_prompter_response(response_text, state["intent"])
        user_cmd = parsed.text_command
        is_complete = parsed.is_interaction_complete

    # Hallucination Check
    potential_hallucination = False
    hallucination_reason = ""

    # Only check if DUT has responded (idx > 0) and router thinks it's done
    if idx > 0 and is_complete:
        if "?" in dut_response_text:
            potential_hallucination = True
            hallucination_reason = "DUT asked a question but router marked complete."
        elif expected_turns > 0 and turn_count < expected_turns:
            potential_hallucination = True
            hallucination_reason = f"Early termination: finished in {turn_count} turns, expected {expected_turns}."

    if potential_hallucination:
        hallucinations += 1
        if VERBOSE:
            print(
                f"    [WARNING] Potential Router Hallucination! {hallucination_reason}"
            )

    # --- Log LLM Call ---
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
        "potential_hallucination": potential_hallucination,
        "hallucination_reason": hallucination_reason,
    }
    current_logs.append(log_entry)

    if VERBOSE:
        print(f"    [DEBUG] Turn {turn_count} | LLM Time: {llm_time:.3f}s")
        if idx > 0:
            print(
                f'           |- DUT Response: "{dut_response_text[:70]}{"..." if len(dut_response_text) > 70 else ""}"'
            )
        print(
            f'           |- Router Decision: is_complete={is_complete} | Next Cmd: "{user_cmd[:60]}{"..." if len(user_cmd) > 60 else ""}"'
        )

    # 2. Simulate DUT Response
    if idx < len(dut_script):
        dut_response = dut_script[idx]
        new_idx = idx + 1
    else:
        dut_response = "I didn't understand."
        new_idx = idx
        is_complete = True

    # 3. Update State & Routing
    tester_msg = HumanMessage(content=user_cmd, name="Tester")
    dut_msg = AIMessage(content=dut_response, name="DUT")

    current_msgs = list(recent_msgs)
    recent_msgs_updated = (current_msgs + [tester_msg, dut_msg])[
        -4:
    ]  # Keep last 2 turns

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
        "hallucination_count": hallucinations,
    }


def judge_node(state: BrainState):
    if llm is None:
        raise RuntimeError("LLM is not initialized.")

    """Evaluates the final interaction against the intent."""
    msgs = state.get("recent_messages", [])
    conversation_log = "\n".join([f"{m.name}: {m.content}" for m in msgs])
    current_logs = state.get("llm_logs", [])

    prompt = f"""You are a QA Judge.
Intent: {state["intent"]}
Language: {state["target_language"]}

Conversation History:
{conversation_log}

Task: Did the DUT successfully fulfill the user's intent?
Output ONLY JSON: {{"verdict": "PASS" or "FAIL", "reasoning": "short explanation"}}
"""

    assert llm_tracker is not None, "LLM Tracker not initialized"

    start_idx = len(llm_tracker.calls)
    response = llm.invoke([HumanMessage(content=prompt)], config={"tags": ["judgment"]})
    end_idx = len(llm_tracker.calls)
    llm_time = sum(c["duration"] for c in llm_tracker.calls[start_idx:end_idx])
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

    if VERBOSE:
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


# ==============================================================================
# 5. GRAPH TOPOLOGY
# ==============================================================================
workflow = StateGraph(BrainState)
workflow.add_node("prompter", prompter_node)
workflow.add_node("judge", judge_node)

workflow.add_edge(START, "prompter")
workflow.add_conditional_edges(
    "prompter",
    lambda state: state["routing_flag"],
    {
        "AWAITING_DUT": "prompter",  # Loop back
        "PROCEED_TO_JUDGE": "judge",  # Evaluation
        "TERMINATED": END,
    },
)
workflow.add_edge("judge", END)

app_graph = workflow.compile()


# ==============================================================================
# 6. EXECUTION & METRIC CALCULATION
# ==============================================================================
def load_test_cases(file_path: str) -> list:
    path = Path(file_path)
    if not path.exists():
        sys.exit(f"Error: {file_path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("test_suite", [])


def run_single_test(test_case: dict) -> dict:
    """Runs a test and calculates the metric externally."""
    expected_turns = test_case.get("expected_turns", 0)

    initial_state: BrainState = {
        "session_id": str(uuid.uuid4()),
        "intent": test_case["intent"],
        "target_language": test_case["language"],
        "recent_messages": [],
        "dut_script": test_case["dut_script"],
        "dut_script_idx": 0,
        "routing_flag": "AWAITING_DUT",
        "final_verdict": "ERROR",
        "final_reasoning": "",
        # Initialize metrics
        "turn_count": 0,
        "expected_turns": expected_turns,
        "llm_logs": [],
        "hallucination_count": 0,
    }

    start_time = time.time()
    try:
        final_state = app_graph.invoke(initial_state, {"recursion_limit": 10})
        elapsed = time.time() - start_time

        model_verdict = final_state["final_verdict"]
        expected_verdict = test_case["expected_verdict"]
        is_correct = model_verdict == expected_verdict

        actual_turns = final_state.get("turn_count", 0)
        turns_match = (actual_turns == expected_turns) if expected_turns > 0 else True

        logs = final_state.get("llm_logs", [])
        total_llm_time = sum(log["llm_time_sec"] for log in logs)
        initial_prompter_time = sum(
            log["llm_time_sec"]
            for log in logs
            if log.get("metric_type") == "initial_prompter"
        )
        routing_voice_cmd_time = sum(
            log["llm_time_sec"]
            for log in logs
            if log.get("metric_type") == "routing_voice_cmd"
        )
        judgment_call_time = sum(
            log["llm_time_sec"] for log in logs if log.get("metric_type") == "judgment"
        )

        return {
            "name": test_case["name"],
            "domain": test_case["domain"],
            "expected": expected_verdict,
            "actual": model_verdict,
            "correct": is_correct,
            "reasoning": final_state["final_reasoning"],
            "time": elapsed,
            "expected_turns": expected_turns,
            "actual_turns": actual_turns,
            "turns_match": turns_match,
            "total_llm_time": total_llm_time,
            "initial_prompter_time": initial_prompter_time,
            "routing_voice_cmd_time": routing_voice_cmd_time,
            "judgment_call_time": judgment_call_time,
            "hallucination_count": final_state.get("hallucination_count", 0),
            "llm_logs": logs,
        }
    except Exception as e:
        return {
            "name": test_case["name"],
            "domain": test_case["domain"],
            "expected": test_case["expected_verdict"],
            "actual": "ERROR",
            "correct": False,
            "reasoning": str(e),
            "time": time.time() - start_time,
            "expected_turns": expected_turns,
            "actual_turns": 0,
            "turns_match": False,
            "total_llm_time": 0,
            "initial_prompter_time": 0,
            "routing_voice_cmd_time": 0,
            "judgment_call_time": 0,
            "hallucination_count": 0,
            "llm_logs": [],
        }


def run_benchmark(cfg: Config):
    test_suite = load_test_cases(TEST_SAMPLES)
    results = []

    print("\n" + "=" * 95)
    print(
        f"AUTOMOTIVE VOICE QA BENCHMARK | Model: {cfg.model_name} | Tests: {len(test_suite)}"
    )
    print("=" * 95)

    for i, tc in enumerate(test_suite, 1):
        print(f"\n[{i}/{len(test_suite)}] Executing: {tc['name']} ({tc['domain']})")
        result = run_single_test(tc)
        results.append(result)

        status = "MATCH" if result["correct"] else "MISMATCH"
        print(
            f"   -> Result: {status} (Expected: {result['expected']}, Got: {result['actual']})"
        )

    generate_report(results, cfg)


def generate_report(results, cfg: Config):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = (correct / total * 100) if total > 0 else 0

    total_expected_turns = sum(r.get("expected_turns", 0) for r in results)
    total_actual_turns = sum(r.get("actual_turns", 0) for r in results)
    turns_matched = sum(1 for r in results if r.get("turns_match", False))
    turn_match_rate = (turns_matched / total * 100) if total > 0 else 0

    total_llm_time = sum(r.get("total_llm_time", 0) for r in results)
    total_initial_prompter_time = sum(
        r.get("initial_prompter_time", 0) for r in results
    )
    total_routing_voice_cmd_time = sum(
        r.get("routing_voice_cmd_time", 0) for r in results
    )
    total_judgment_call_time = sum(r.get("judgment_call_time", 0) for r in results)
    total_hallucinations = sum(r.get("hallucination_count", 0) for r in results)

    print("\n" + "=" * 95)
    print("FINAL BENCHMARK REPORT")
    print("=" * 95)

    # ASCII Table
    header = f"{'Test Name':<25} {'Domain':<10} {'Exp':<4} {'Act':<4} {'Stat':<5} {'ExpT':<4} {'ActT':<4} {'InitT':<6} {'RouteT':<6} {'JudgeT':<6}"
    print(header)
    print("-" * 95)

    for r in results:
        status = "PASS" if r["correct"] else "FAIL"
        print(
            f"{r['name']:<25} {r['domain']:<10} {r['expected']:<4} {r['actual']:<4} {status:<5} {r.get('expected_turns', 0):<4} {r.get('actual_turns', 0):<4} {r.get('initial_prompter_time', 0):<6.2f} {r.get('routing_voice_cmd_time', 0):<6.2f} {r.get('judgment_call_time', 0):<6.2f}"
        )

    print("-" * 95)
    print(f"VERDICT ACCURACY: {accuracy:.1f}% ({correct}/{total})")
    print(f"TURN MATCH RATE: {turn_match_rate:.1f}% ({turns_matched}/{total})")
    print(
        f"AVG EXPECTED TURNS: {total_expected_turns/total:.1f} | AVG ACTUAL TURNS: {total_actual_turns/total:.1f}"
    )
    print(f"TOTAL LLM TIME: {total_llm_time:.2f}s")
    print(f"  - INITIAL PROMPTER: {total_initial_prompter_time:.2f}s")
    print(f"  - ROUTING + VOICE CMD: {total_routing_voice_cmd_time:.2f}s")
    print(f"  - JUDGMENT CALL: {total_judgment_call_time:.2f}s")
    print(f"TOTAL ROUTER HALLUCINATIONS: {total_hallucinations}")
    print("=" * 95 + "\n")

    # Save JSON
    report_file = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Include raw callback data for absolute transparency
    raw_llm_calls = llm_tracker.calls if llm_tracker is not None else []

    with open(report_file, "w") as f:
        json.dump(
            {
                "model": cfg.model_name,
                "accuracy": accuracy,
                "turn_match_rate": turn_match_rate,
                "avg_expected_turns": total_expected_turns / total if total > 0 else 0,
                "avg_actual_turns": total_actual_turns / total if total > 0 else 0,
                "total_llm_time": total_llm_time,
                "total_initial_prompter_time": total_initial_prompter_time,
                "total_routing_voice_cmd_time": total_routing_voice_cmd_time,
                "total_judgment_call_time": total_judgment_call_time,
                "total_hallucinations": total_hallucinations,
                "raw_llm_callback_logs": raw_llm_calls,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Detailed report saved to {report_file}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="conversation-eval-pipeline")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="the name or identifier of the model under test.",
    )

    parser.add_argument(
        "--test_samples",
        type=str,
        default="test_cases.json",
        help="the config file path ",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="the port which the llama server is listening",
    )

    args = parser.parse_args()

    global TEST_SAMPLES, VERBOSE, llm, llm_tracker

    TEST_SAMPLES = args.test_samples
    cfg = Config(
        model_name=args.model_name,
        port=args.port,
        test_samples=args.test_samples,
        verbose=args.verbose,
    )

    llm_tracker = LLMTimeTracker()
    llm = ChatOpenAI(**cfg.llm_config, callbacks=[llm_tracker])
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
