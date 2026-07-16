import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from graph import ConversationGraph
from llm_tracker import LLMTimeTracker
from reporter import generate_report
from runner import run_benchmark
from langchain_openai import ChatOpenAI


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

    cfg = Config(
        model_name=args.model_name,
        port=args.port,
        test_samples=args.test_samples,
        verbose=args.verbose,
    )

    tracker = LLMTimeTracker()
    llm = ChatOpenAI(**cfg.llm_config, callbacks=[tracker])

    graph = ConversationGraph(llm, verbose=cfg.verbose)

    results = run_benchmark(cfg, graph)
    generate_report(results, cfg)


if __name__ == "__main__":
    main()
