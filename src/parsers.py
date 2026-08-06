import json
import re

from schemas import JudgeSchema, PrompterSchema


def extract_json_from_text(text: str) -> dict | None:
    """JSON extraction"""
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    if '"text_command"' in text or '"verdict"' in text:
        try:
            cleaned = re.sub(r",\s*}", "}", text)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def parse_prompter_response(text: str, default_intent: str) -> PrompterSchema:
    json_data = extract_json_from_text(text)
    if json_data and "text_command" in json_data:
        return PrompterSchema(text_command=json_data["text_command"])
    clean_text = text.strip().replace('"', "").split("\n")[0]
    return PrompterSchema(text_command=clean_text or default_intent)


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
