from fastapi import FastAPI, UploadFile, File, status, HTTPException
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader
from vector_store import add_document_to_vectorstore
from typing import List

import tempfile, os
from models import DocumentUploadResponse, AgentResponse, QueryRequest, TraceEvent
from langchain_core.messages import HumanMessage, AIMessage
from agent import rag_agent

app = FastAPI(
    title="LangGraph RAG Agent API",
    description="API for Langgraph powered RAG with pinecone and Groq",
    version="1.0.0"
)

memory = MemorySaver()

@app.post("/upload-document/",response_model=DocumentUploadResponse, status_code=status.HTTP_200_OK)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads PDF document, extracts text, and add its to RAG knowledge base.
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail = "Only PDF files are supported"
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        file_content = await file.read()
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name

    print(f"Received PDF for upload: {file.filename}. Saved temporarily to {tmp_file_path}")

    try:
        loader = PyPDFLoader(tmp_file_path)
        document = loader.load()

        total_chunks_added = 0

        if document:
            full_text_content = "\n\n".join([doc.page_content for doc in document])
            add_document_to_vectorstore(full_text_content)
            total_chunks_added = len(document)

        return DocumentUploadResponse(
            message=f"PDF {file.filename} successfully uploaded and indexed",
            filename = file.filename,
            processed_chunks=total_chunks_added
        )
    
    except Exception as e:
        print(f"Error processing PDF document : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to process PDF : {e}"
        )
    
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            print(f"Cleaned up temporary file: {tmp_file_path}")

@app.post("/chat/", response_model=AgentResponse)
async def chat_with_agents(request: QueryRequest):
    trace_events_for_frontend : List[TraceEvent] = []

    try:
        config = {
            "configurable" : {
                "thread_id" : request.session_id,
                "web_search_enabled" : request.enable_web_search
            }
        }

        inputs = {"messages" : [HumanMessage(content=request.query)] }

        final_message = ""

        print(f"--- Starting Agent Stream for session {request.session_id} ---")
        print(f"Web Search Enabled: {request.enable_web_search}") # For server-side debugging

        for i, s in enumerate(rag_agent.stream(inputs, config=config)):
            current_node_name = None
            node_output_state = None

        final_actual_state_dict = None
        if s:
            if "__end__" in s:
                final_actual_state_dict = s["__end__"]
            else:
                if list(s.keys()):
                    final_actual_state_dict = s[list(s.keys())[0]]

        
        if final_actual_state_dict and "messages" in final_actual_state_dict:
            for msg in reversed(final_actual_state_dict["messages"]):
                if isinstance(msg, AIMessage):
                    final_message = msg.content
                    break

        if not final_message:
             print("Agent finished, but no final AIMessage found in the final state after stream completion.")
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Agent did not return a valid response (final AI message not found).")

        print(f"--- Agent Stream Ended. Final Response: {final_message[:200]}... ---")

        return AgentResponse(response=final_message)

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_details = f"Error during agent invocation: {e}"
        print(error_details)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal Server Error: {e}")