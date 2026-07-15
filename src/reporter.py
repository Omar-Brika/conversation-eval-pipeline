import json
from datetime import datetime

from config import Config


def generate_report(results: list, cfg: Config):
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

    total_llm_calls = sum(r.get("total_llm_calls", 0) for r in results)
    total_initial_calls = sum(r.get("initial_prompter_calls", 0) for r in results)
    total_routing_calls = sum(r.get("routing_voice_cmd_calls", 0) for r in results)
    total_judgment_calls = sum(r.get("judgment_call_calls", 0) for r in results)

    # Calculate Averages
    avg_total_llm_time = (
        (total_llm_time / total_llm_calls) if total_llm_calls > 0 else 0
    )
    avg_initial_prompter_time = (
        (total_initial_prompter_time / total_initial_calls)
        if total_initial_calls > 0
        else 0
    )
    avg_routing_voice_cmd_time = (
        (total_routing_voice_cmd_time / total_routing_calls)
        if total_routing_calls > 0
        else 0
    )
    avg_judgment_call_time = (
        (total_judgment_call_time / total_judgment_calls)
        if total_judgment_calls > 0
        else 0
    )

    print("\n" + "=" * 95)
    print("FINAL BENCHMARK REPORT")
    print("=" * 95)

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

    print(
        f"TOTAL LLM TIME: {total_llm_time:.2f}s (Avg: {avg_total_llm_time:.3f}s/call)"
    )
    print(
        f"  - INITIAL PROMPTER: {total_initial_prompter_time:.2f}s (Avg: {avg_initial_prompter_time:.3f}s/call)"
    )
    print(
        f"  - ROUTING + VOICE CMD: {total_routing_voice_cmd_time:.2f}s (Avg: {avg_routing_voice_cmd_time:.3f}s/call)"
    )
    print(
        f"  - JUDGMENT CALL: {total_judgment_call_time:.2f}s (Avg: {avg_judgment_call_time:.3f}s/call)"
    )
    print("=" * 95 + "\n")

    report_file = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    clean_results = [{k: v for k, v in r.items() if k != "llm_logs"} for r in results]

    with open(report_file, "w") as f:
        json.dump(
            {
                "model": cfg.model_name,
                "accuracy": accuracy,
                "turn_match_rate": turn_match_rate,
                "avg_expected_turns": total_expected_turns / total if total > 0 else 0,
                "avg_actual_turns": total_actual_turns / total if total > 0 else 0,
                # Totals
                "total_llm_time": total_llm_time,
                "total_initial_prompter_time": total_initial_prompter_time,
                "total_routing_voice_cmd_time": total_routing_voice_cmd_time,
                "total_judgment_call_time": total_judgment_call_time,
                # Averages
                "avg_total_llm_time": avg_total_llm_time,
                "avg_initial_prompter_time": avg_initial_prompter_time,
                "avg_routing_voice_cmd_time": avg_routing_voice_cmd_time,
                "avg_judgment_call_time": avg_judgment_call_time,
                "results": clean_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Detailed report saved to {report_file}")
