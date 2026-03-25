from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ExecuteRequest(BaseModel):
    code: str = Field(..., max_length=50_000, description="Python code to execute")
    timeout: int = Field(default=30, ge=1, le=60, description="Max execution time in seconds")


class ExecuteResponse(BaseModel):
    output: str
    error: Optional[str] = None
    execution_time_ms: int
    variables: Dict[str, str] = {}
    plots: List[str] = []


class CodeTemplate(BaseModel):
    id: str
    name: str
    description: str
    code: str
    category: str


class HistoryItem(BaseModel):
    code: str
    output: str
    error: Optional[str] = None
    execution_time_ms: int
    timestamp: str


class SuggestRequest(BaseModel):
    prompt: str = Field(..., max_length=1000, description="Description of what to code")
    context: str = Field(default="", max_length=10_000, description="Current editor code for context")


class SuggestResponse(BaseModel):
    suggestion: str
