from typing import Dict, List, Literal, TypedDict, Any
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# RAG models

class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    route: Literal["rag", "web", "answer", "end"]
    rag : str
    web : str
    web_searched_enabled : bool

class RouteDecision(BaseModel):
    route: Literal["rag", "web", "answer", "end"]
    reply: str | None = Field(None, description="Filled only when route == end")

class RagJudge(BaseModel):
    sufficient : bool = Field(..., description="True if retired information is sufficient to answer the user's question, False otherwise.")


# API models

class TraceEvent(BaseModel):
    step : int
    node_name : str
    description : str
    details : Dict[str, Any] = Field(default_factory=dict)
    event_type : str

class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    processed_chunks: int

class AgentResponse(BaseModel):
    response : str
    # trace_events : List[TraceEvent] = Field(default_factory=list)

class QueryRequest(BaseModel):
    session_id : str
    query: str
    enable_web_search : bool = True