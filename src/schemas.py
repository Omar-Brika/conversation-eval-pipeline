from typing import Literal

from pydantic import BaseModel, Field


class PrompterSchema(BaseModel):
    text_command: str = Field(
        description="Exact voice command to send to the DUT. Empty string if the interaction is complete."
    )


class JudgeSchema(BaseModel):
    reasoning: str = Field(description="Brief evaluation of semantic equivalence.")
    verdict: Literal["PASS", "FAIL"]