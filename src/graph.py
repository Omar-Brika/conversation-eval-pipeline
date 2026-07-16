from typing import Callable

from langchain_core.callbacks import BaseCallbackManager
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from llm_tracker import LLMTimeTracker
from nodes import create_judge_node, create_prompter_node
from state import BrainState


def _build_graph(
    prompter_node: Callable, judge_node: Callable
) -> CompiledStateGraph:
    workflow = StateGraph(BrainState)
    workflow.add_node("prompter", prompter_node)
    workflow.add_node("judge", judge_node)

    workflow.add_edge(START, "prompter")
    workflow.add_conditional_edges(
        "prompter",
        lambda state: state["routing_flag"],
        {
            "AWAITING_DUT": "prompter",
            "PROCEED_TO_JUDGE": "judge",
            "TERMINATED": END,
        },
    )
    workflow.add_edge("judge", END)

    return workflow.compile()


class ConversationGraph:
    def __init__(self, llm: ChatOpenAI, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        self.tracker = self._extract_tracker(llm)
        self._graph = self._build()

    @staticmethod
    def _extract_tracker(llm: ChatOpenAI) -> LLMTimeTracker:
        callbacks = llm.callbacks

        if callbacks is None:
            tracker = LLMTimeTracker()
            llm.callbacks = [tracker]
            return tracker

        if isinstance(callbacks, BaseCallbackManager):
            for h in callbacks.handlers:
                if isinstance(h, LLMTimeTracker):
                    return h
            tracker = LLMTimeTracker()
            callbacks.add_handler(tracker)
            return tracker

        for h in callbacks:
            if isinstance(h, LLMTimeTracker):
                return h
        tracker = LLMTimeTracker()
        callbacks.append(tracker)
        return tracker

    def _build(self) -> CompiledStateGraph:
        prompter_node = create_prompter_node(self.llm, self.verbose)
        judge_node = create_judge_node(self.llm, self.verbose)
        return _build_graph(prompter_node, judge_node)

    def invoke(self, initial_state: BrainState) -> dict:
        return self._graph.invoke(initial_state, {"recursion_limit": 10})
