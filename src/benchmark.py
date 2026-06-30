import uuid
import json
import sys
import re
import time
from pathlib import Path
from typing import Sequence, TypedDict, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG_FILE = "test_cases.json"
VERBOSE = True  

LLM_CONFIG = {
    "base_url": "http://localhost:8000/v1",  
    "api_key": "notneeded",
    "model": "qwen3.5-4b",                  
    "temperature": 0.0,
    "max_retries": 2,
    "max_tokens": 512,
}

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
# 3. MODEL INITIALIZATION
# ==============================================================================
llm = ChatOpenAI(**LLM_CONFIG)

# ==============================================================================
# 4. JSON PARSING 
# ==============================================================================
def extract_json_from_text(text: str) -> dict | None:
    """JSON extraction"""
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try: return json.loads(json_match.group(1))
        except: pass
    
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try: return json.loads(json_match.group(0))
        except: pass
        
    if '"text_command"' in text or '"verdict"' in text:
        try:
            cleaned = re.sub(r',\s*}', '}', text)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match: return json.loads(match.group(0))
        except: pass
        
    return None

def parse_prompter_response(text: str, default_intent: str) -> PrompterSchema:
    json_data = extract_json_from_text(text)
    if json_data and "text_command" in json_data:
        return PrompterSchema(
            text_command=json_data["text_command"],
            is_interaction_complete=bool(json_data.get("is_interaction_complete", False))
        )
    
    clean_text = text.strip().replace('"', '').split('\n')[0]
    return PrompterSchema(text_command=clean_text or default_intent, is_interaction_complete=False)

def parse_judge_response(text: str) -> JudgeSchema:
    json_data = extract_json_from_text(text)
    if json_data and "verdict" in json_data:
        verdict_raw = str(json_data.get("verdict", "FAIL")).upper()
        verdict = "PASS" if "PASS" in verdict_raw else "FAIL"
        return JudgeSchema(verdict=verdict, reasoning=json_data.get("reasoning", "No reasoning"))

    text_lower = text.lower()
    verdict = "FAIL"
    if "pass" in text_lower and "fail" not in text_lower:
        verdict = "PASS"
    
    return JudgeSchema(verdict=verdict, reasoning=text.strip()[:150])

# ==============================================================================
# 5. NODE DEFINITIONS
# ==============================================================================
def prompter_node(state: BrainState):
    """
    Acts as the Decision Engine. Generates commands, simulates DUT, 
    and decides whether to loop or proceed to Judge.
    """
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
        sys_prompt = SystemMessage(content=(
            f"You are a Test User. Language: {state['target_language']}. "
            f"Intent: {state['intent']}. Generate ONLY the natural voice command. "
            f"Output JSON: {{\"text_command\": \"...\", \"is_interaction_complete\": false}}"
        ))
        
        start_llm = time.time()
        response = llm.invoke([sys_prompt, HumanMessage(content="Generate.")])
        llm_time = time.time() - start_llm
        
        parsed = parse_prompter_response(response.content, state['intent'])
        user_cmd = parsed.text_command
        is_complete = False
        dut_response_text = "N/A (First Turn)"
    else:
        # Turn 2+: Evaluate last DUT response and generate next command
        last_dut_msg = recent_msgs[-1].content
        dut_response_text = last_dut_msg
        
        sys_prompt = SystemMessage(content=(
            f"You are a Test User. Intent: {state['intent']}. Language: {state['target_language']}.\n"
            f"The Assistant just said: \"{last_dut_msg}\"\n"
            f"Did the assistant complete the task or refuse? If yes, set is_interaction_complete to true.\n"
            f"If the assistant asked for clarification, set it to false and generate the next command.\n"
            f"Output JSON: {{\"text_command\": \"...\", \"is_interaction_complete\": true/false}}"
        ))
        
        start_llm = time.time()
        response = llm.invoke([sys_prompt, HumanMessage(content="Evaluate and reply.")])
        llm_time = time.time() - start_llm
        
        parsed = parse_prompter_response(response.content, state['intent'])
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
            print(f"    [WARNING] Potential Router Hallucination! {hallucination_reason}")

    # --- Log LLM Call ---
    log_entry = {
        "node": "prompter",
        "turn": turn_count,
        "llm_time_sec": round(llm_time, 4),
        "dut_response": dut_response_text,
        "raw_llm_output": response.content,
        "parsed_decision": {
            "text_command": parsed.text_command,
            "is_interaction_complete": parsed.is_interaction_complete
        },
        "potential_hallucination": potential_hallucination,
        "hallucination_reason": hallucination_reason
    }
    current_logs.append(log_entry)
    
    if VERBOSE:
        print(f"    [DEBUG] Turn {turn_count} | LLM Time: {llm_time:.3f}s")
        if idx > 0:
            print(f"           |- DUT Response: \"{dut_response_text[:70]}{'...' if len(dut_response_text) > 70 else ''}\"")
        print(f"           |- Router Decision: is_complete={is_complete} | Next Cmd: \"{user_cmd[:60]}{'...' if len(user_cmd) > 60 else ''}\"")

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
    recent_msgs_updated = (current_msgs + [tester_msg, dut_msg])[-4:] # Keep last 2 turns

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
        "hallucination_count": hallucinations
    }

def judge_node(state: BrainState):
    """Evaluates the final interaction against the intent."""
    msgs = state.get("recent_messages", [])
    conversation_log = "\n".join([f"{m.name}: {m.content}" for m in msgs])
    current_logs = state.get("llm_logs", [])
    
    prompt = f"""You are a QA Judge.
Intent: {state['intent']}
Language: {state['target_language']}

Conversation History:
{conversation_log}

Task: Did the DUT successfully fulfill the user's intent?
Output ONLY JSON: {{"verdict": "PASS" or "FAIL", "reasoning": "short explanation"}}
"""
    start_llm = time.time()
    response = llm.invoke([HumanMessage(content=prompt)])
    llm_time = time.time() - start_llm
    
    parsed = parse_judge_response(response.content)

    log_entry = {
        "node": "judge",
        "llm_time_sec": round(llm_time, 4),
        "raw_llm_output": response.content,
        "parsed_decision": {
            "verdict": parsed.verdict,
            "reasoning": parsed.reasoning
        }
    }
    current_logs.append(log_entry)
    
    if VERBOSE:
        print(f"    [DEBUG] Judge Evaluation | LLM Time: {llm_time:.3f}s")
        print(f"           |- Verdict: {parsed.verdict} | Reasoning: \"{parsed.reasoning[:70]}{'...' if len(parsed.reasoning) > 70 else ''}\"")

    return {
        "routing_flag": "TERMINATED",
        "final_verdict": parsed.verdict,
        "final_reasoning": parsed.reasoning,
        "llm_logs": current_logs
    }

# ==============================================================================
# 6. GRAPH TOPOLOGY
# ==============================================================================
workflow = StateGraph(BrainState)
workflow.add_node("prompter", prompter_node)
workflow.add_node("judge", judge_node)

workflow.add_edge(START, "prompter")
workflow.add_conditional_edges(
    "prompter",
    lambda state: state["routing_flag"],
    {
        "AWAITING_DUT": "prompter",       # Loop back
        "PROCEED_TO_JUDGE": "judge",      # Evaluation
        "TERMINATED": END
    }
)
workflow.add_edge("judge", END)

app_graph = workflow.compile()

# ==============================================================================
# 7. EXECUTION & METRIC CALCULATION
# ==============================================================================
def load_test_cases(file_path: str) -> list:
    path = Path(file_path)
    if not path.exists(): sys.exit(f"Error: {file_path} not found.")
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
        "hallucination_count": 0
    }

    start_time = time.time()
    try:
        final_state = app_graph.invoke(initial_state, {"recursion_limit": 10})
        elapsed = time.time() - start_time
        
        model_verdict = final_state["final_verdict"]
        expected_verdict = test_case["expected_verdict"] 
        is_correct = (model_verdict == expected_verdict)
        
        actual_turns = final_state.get("turn_count", 0)
        turns_match = (actual_turns == expected_turns) if expected_turns > 0 else True
        
        total_llm_time = sum(log["llm_time_sec"] for log in final_state.get("llm_logs", []))
        
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
            "hallucination_count": final_state.get("hallucination_count", 0),
            "llm_logs": final_state.get("llm_logs", [])
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
            "hallucination_count": 0,
            "llm_logs": []
        }

def run_benchmark():
    test_suite = load_test_cases(CONFIG_FILE)
    results = []
    
    print("\n" + "="*95)
    print(f"AUTOMOTIVE VOICE QA BENCHMARK | Model: {LLM_CONFIG['model']} | Tests: {len(test_suite)}")
    print("="*95)

    for i, tc in enumerate(test_suite, 1):
        print(f"\n[{i}/{len(test_suite)}] Executing: {tc['name']} ({tc['domain']})")
        result = run_single_test(tc)
        results.append(result)
        
        status = "MATCH" if result["correct"] else "MISMATCH"
        print(f"   -> Result: {status} (Expected: {result['expected']}, Got: {result['actual']})")

    generate_report(results)

def generate_report(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    total_expected_turns = sum(r.get("expected_turns", 0) for r in results)
    total_actual_turns = sum(r.get("actual_turns", 0) for r in results)
    turns_matched = sum(1 for r in results if r.get("turns_match", False))
    turn_match_rate = (turns_matched / total * 100) if total > 0 else 0
    
    total_llm_time = sum(r.get("total_llm_time", 0) for r in results)
    total_hallucinations = sum(r.get("hallucination_count", 0) for r in results)
    
    print("\n" + "="*95)
    print("FINAL BENCHMARK REPORT")
    print("="*95)
    
    # ASCII Table
    header = f"{'Test Name':<30} {'Domain':<12} {'Exp':<5} {'Act':<5} {'Status':<8} {'ExpT':<5} {'ActT':<5} {'Time':<6}"
    print(header)
    print("-" * 95)
    
    for r in results:
        status = "PASS" if r["correct"] else "FAIL"
        print(f"{r['name']:<30} {r['domain']:<12} {r['expected']:<5} {r['actual']:<5} {status:<8} {r.get('expected_turns', 0):<5} {r.get('actual_turns', 0):<5} {r['time']:<6.2f}")
        
    print("-" * 95)
    print(f"VERDICT ACCURACY: {accuracy:.1f}% ({correct}/{total})")
    print(f"TURN MATCH RATE: {turn_match_rate:.1f}% ({turns_matched}/{total})")
    print(f"AVG EXPECTED TURNS: {total_expected_turns/total:.1f} | AVG ACTUAL TURNS: {total_actual_turns/total:.1f}")
    print(f"TOTAL LLM TIME: {total_llm_time:.2f}s")
    print(f"TOTAL ROUTER HALLUCINATIONS: {total_hallucinations}")
    print("="*95 + "\n")
    
    # Save JSON
    report_file = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump({
            "model": LLM_CONFIG["model"], 
            "accuracy": accuracy, 
            "turn_match_rate": turn_match_rate,
            "avg_expected_turns": total_expected_turns/total if total > 0 else 0,
            "avg_actual_turns": total_actual_turns/total if total > 0 else 0,
            "total_llm_time": total_llm_time,
            "total_hallucinations": total_hallucinations,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Detailed report saved to {report_file}")

if __name__ == "__main__":
    run_benchmark()
