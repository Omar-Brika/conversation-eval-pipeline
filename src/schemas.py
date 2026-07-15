from typing import Literal

from pydantic import BaseModel, Field


class PrompterSchema(BaseModel):
    text_command: str = Field(description="Exact voice command to send to the DUT.")
    is_interaction_complete: bool = Field(
        description="True if the DUT's response concludes the interaction (task done, or explicitly failed). False if DUT asked for clarification."
    )


class JudgeSchema(BaseModel):
    reasoning: str = Field(description="Brief evaluation of semantic equivalence.")
    verdict: Literal["PASS", "FAIL"]
