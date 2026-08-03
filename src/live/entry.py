import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI

from benchmark import LiveBenchmark, make_mock_dut_speech_fn, mock_voice_command_fn


def _resolve_test_samples(path: str) -> str:
    candidates = [
        Path(path),
        Path(__file__).resolve().parent.parent.parent / path,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    sys.exit(f"Error: test samples file not found: {path}")


def load_test_cases(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("test_suite", [])


def main():
    parser = argparse.ArgumentParser(
        description="Live benchmark entry point (wiring test with mock TTS/ASR)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="the name or identifier of the model under test.",
    )
    parser.add_argument(
        "--test_samples",
        type=str,
        default="Json_samples/samples.json",
        help="path to a JSON test suite (default: Json_samples/samples.json)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="the port which the llama server is listening",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="max number of test cases to run (default: 1)",
    )
    args = parser.parse_args()

    llm = ChatOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="notneeded",
        model=args.model_name,
        temperature=0.0,
        max_retries=2,
        max_tokens=512,
    )

    test_cases = load_test_cases(_resolve_test_samples(args.test_samples))
    if args.limit > 0:
        test_cases = test_cases[: args.limit]

    print("\n" + "=" * 95)
    print(
        f"LIVE BENCHMARK WIRING TEST | Model: {args.model_name} | Cases: {len(test_cases)}"
    )
    print("=" * 95)

    results = []
    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Executing: {tc.get('name', f'test_{i}')} ({tc.get('domain', '')})")
        speech_fn = make_mock_dut_speech_fn(tc.get("dut_script"))
        bench = LiveBenchmark(llm, mock_voice_command_fn, speech_fn)
        result = bench.run_test_case(tc)
        results.append(result)
        status = "MATCH" if result["correct"] else "MISMATCH"
        print(
            f"   -> Result: {status} (Expected: {result['expected']}, Got: {result['actual']}) "
            f"| time: {result['time']:.2f}s | turns: {result['actual_turns']} "
            f"| llm: {result['total_llm_time']:.2f}s"
        )

    correct = sum(1 for r in results if r["correct"])
    print("\n" + "=" * 95)
    print(f"FINAL: {correct}/{len(results)} correct")
    print("=" * 95)


if __name__ == "__main__":
    main()
