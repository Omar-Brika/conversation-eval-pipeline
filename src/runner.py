import json
import sys
import time
import uuid
from pathlib import Path

from config import Config
from state import BrainState


def load_test_cases(file_path: str) -> list:
    path = Path(file_path)
    if not path.exists():
        sys.exit(f"Error: {file_path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("test_suite", [])


def run_single_test(test_case: dict, app_graph) -> dict:
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
        "turn_count": 0,
        "expected_turns": expected_turns,
        "llm_logs": [],
    }

    start_time = time.time()
    try:
        final_state = app_graph.invoke(initial_state)
        elapsed = time.time() - start_time

        model_verdict = final_state["final_verdict"]
        expected_verdict = test_case["expected_verdict"]
        is_correct = model_verdict == expected_verdict

        actual_turns = final_state.get("turn_count", 0)
        turns_match = (actual_turns == expected_turns) if expected_turns > 0 else True

        logs = final_state.get("llm_logs", [])
        # calculate call counts
        total_calls = len(logs)
        initial_calls = sum(
            1 for log in logs if log.get("metric_type") == "initial_prompter"
        )
        routing_calls = sum(
            1 for log in logs if log.get("metric_type") == "routing_voice_cmd"
        )
        judgment_calls = sum(1 for log in logs if log.get("metric_type") == "judgment")

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
            "total_llm_calls": total_calls,
            "initial_prompter_calls": initial_calls,
            "routing_voice_cmd_calls": routing_calls,
            "judgment_call_calls": judgment_calls,
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
            "total_llm_calls": 0,
            "initial_prompter_calls": 0,
            "routing_voice_cmd_calls": 0,
            "judgment_call_calls": 0,
            "llm_logs": [],
        }


def run_benchmark(cfg: Config, app_graph):
    test_suite = load_test_cases(cfg.test_samples)
    results = []

    print("\n" + "=" * 95)
    print(
        f"AUTOMOTIVE VOICE QA BENCHMARK | Model: {cfg.model_name} | Tests: {len(test_suite)}"
    )
    print("=" * 95)

    for i, tc in enumerate(test_suite, 1):
        print(f"\n[{i}/{len(test_suite)}] Executing: {tc['name']} ({tc['domain']})")
        result = run_single_test(tc, app_graph)
        results.append(result)

        status = "MATCH" if result["correct"] else "MISMATCH"
        print(
            f"   -> Result: {status} (Expected: {result['expected']}, Got: {result['actual']})"
        )

    return results
