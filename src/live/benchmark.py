import time
from typing import Callable

from langchain_openai import ChatOpenAI
from live_graph import LiveConversationGraph


def mock_voice_command_fn(text: str) -> bytes:
    return text.encode("utf-8")


def make_mock_dut_speech_fn(dut_script: list[str] | None = None) -> Callable[[bytes], str]:
    """ASR mock that simulates a scripted DUT from a JSON test sample.

    Each call replays the next line of the test case's ``dut_script`` so the
    live graph (prompter -> voice_cmd -> asr -> router) behaves like the
    offline scripted pipeline. When the script is exhausted, the last line is
    repeated so the router can still terminate the conversation loop.
    """
    script = list(dut_script or ["I have completed your request."])
    idx = {"value": 0}

    def mock_speech_recognition_fn(audio: bytes) -> str:
        line = script[min(idx["value"], len(script) - 1)]
        idx["value"] += 1
        return line

    return mock_speech_recognition_fn


class LiveBenchmark:
    def __init__(
        self,
        llm: ChatOpenAI,
        voice_command_fn: Callable[[str], bytes],
        speech_recognition_fn: Callable[[bytes], str],
    ):
        self.llm = llm
        self.graph = LiveConversationGraph(
            llm, voice_command_fn, speech_recognition_fn
        )

    def run_test_case(self, test_case: dict) -> dict:
        start_time = time.time()
        try:
            final_state = self.graph.invoke(
                test_case["intent"], test_case["language"]
            )
            elapsed = time.time() - start_time

            model_verdict = final_state["final_verdict"]
            expected_verdict = test_case.get("expected_verdict", "PASS")
            is_correct = model_verdict == expected_verdict

            logs = list(final_state.get("llm_logs", []))
            total_calls = len(logs)
            initial_prompter_calls = sum(
                1 for l in logs if l.get("metric_type") == "initial_prompter"
            )
            routing_voice_cmd_calls = sum(
                1 for l in logs if l.get("metric_type") == "routing_voice_cmd"
            )
            judgment_calls = sum(1 for l in logs if l.get("metric_type") == "judgment")

            total_llm_time = sum(l["llm_time_sec"] for l in logs)
            initial_prompter_time = sum(
                l["llm_time_sec"] for l in logs if l.get("metric_type") == "initial_prompter"
            )
            routing_voice_cmd_time = sum(
                l["llm_time_sec"] for l in logs if l.get("metric_type") == "routing_voice_cmd"
            )
            judgment_time = sum(l["llm_time_sec"] for l in logs if l.get("metric_type") == "judgment")

            return {
                "name": test_case.get("name", "unnamed"),
                "domain": test_case.get("domain", ""),
                "expected": expected_verdict,
                "actual": model_verdict,
                "correct": is_correct,
                "reasoning": final_state.get("final_reasoning", ""),
                "time": elapsed,
                "actual_turns": final_state.get("turn_count", 0),
                "total_llm_time": total_llm_time,
                "initial_prompter_time": initial_prompter_time,
                "routing_voice_cmd_time": routing_voice_cmd_time,
                "judgment_time": judgment_time,
                "total_llm_calls": total_calls,
                "initial_prompter_calls": initial_prompter_calls,
                "routing_voice_cmd_calls": routing_voice_cmd_calls,
                "judgment_calls": judgment_calls,
                "llm_logs": logs,
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "name": test_case.get("name", "unnamed"),
                "domain": test_case.get("domain", ""),
                "expected": test_case.get("expected_verdict", "PASS"),
                "actual": "ERROR",
                "correct": False,
                "reasoning": str(e),
                "time": elapsed,
                "actual_turns": 0,
                "total_llm_time": 0,
                "prompter_time": 0,
                "router_time": 0,
                "judgment_time": 0,
                "total_llm_calls": 0,
                "prompter_calls": 0,
                "router_calls": 0,
                "judgment_calls": 0,
                "llm_logs": [],
            }

    def run_benchmark(self, test_cases: list[dict]) -> list[dict]:
        results = []
        total = len(test_cases)
        print("\n" + "=" * 95)
        print(f"LIVE BENCHMARK | Model: {self.llm.model} | Tests: {total}")
        print("=" * 95)

        for i, tc in enumerate(test_cases, 1):
            name = tc.get("name", f"test_{i}")
            domain = tc.get("domain", "")
            print(f"\n[{i}/{total}] Executing: {name} ({domain})")
            result = self.run_test_case(tc)
            results.append(result)
            status = "MATCH" if result["correct"] else "MISMATCH"
            print(f"   -> Result: {status} (Expected: {result['expected']}, Got: {result['actual']})")

        return results
