from langchain_core.messages import  HumanMessage, AIMessage
from langchain_groq import ChatGroq
from config import GROQ_MODEL
from langchain_core.runnables import RunnableConfig
from models import RouteDecision, RagJudge, AgentState


router_llm  = ChatGroq(model=GROQ_MODEL, temperature=0).with_structured_output(RouteDecision)
judge_llm  = ChatGroq(model=GROQ_MODEL, temperature=0).with_structured_output(RagJudge)
answer_llm  = ChatGroq(model=GROQ_MODEL, temperature=0.7)

def routernode(state: AgentState, config:RunnableConfig ) -> AgentState:
    print("---- Entering router node -----")
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            query = next(m.content)
        else:
            query = ""

    web_search_enabled = config.get("configurable",{}).get("web_search_enabled", True)
    print(f"Router receieved web search info: {web_search_enabled}")

    system_prompt = (
        "You are an intelligent routing agent designed to direct user queries to the most appropriate tool."
        "Your primary goal is to provide accurate and relevant information by selecting the best source."
        "Prioritize using the **internal knowledge base (RAG)** for factual information that is likely "
        "to be contained within pre-uploaded documents or for common, well-established facts."
    )
    
    if web_search_enabled:
        system_prompt += (
            "You **CAN** use web search for queries that require very current, real-time, or broad general knowledge "
            "that is unlikely to be in a specific, static knowledge base (e.g., today's news, live data, very recent events)."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection (e.g., 'What is X?', 'How does Y work?', 'Explain Z policy')."
            "\n- 'web': For queries about current events, live data, very recent news, or broad general knowledge that requires up-to-date internet access (e.g., 'Who won the election yesterday?', 'What is the weather in London?', 'Latest news on technology')."
        )
    else:
        system_prompt += (
            "**Web search is currently DISABLED.** You **MUST NOT** choose the 'web' route."
            "If a query would normally require web search, you should attempt to answer it using RAG (if applicable) or directly from your general knowledge."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection, AND for queries that would normally go to web search but web search is disabled."
            "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
        )

    system_prompt += (
        "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
        "\n- 'end': For pure greetings or small-talk where no factual answer is expected (e.g., 'Hi', 'How are you?'). If choosing 'end', you MUST provide a 'reply'."
        "\n\nExample routing decisions:"
        "\n- User: 'What are the treatment of diabetes?' -> Route: 'rag' (Factual knowledge, likely in KB)."
        "\n- User: 'What is the capital of France?' -> Route: 'rag' (Common knowledge, can be in KB or answered directly if LLM knows)."
        "\n- User: 'Who won the NBA finals last night?' -> Route: 'web' (Current event, requires live data)."
        "\n- User: 'How do I submit an expense report?' -> Route: 'rag' (Internal procedure)."
        "\n- User: 'Tell me about quantum computing.' -> Route: 'rag' (Foundational knowledge can be in KB. If KB is sparse, judge will route to web if enabled)."
        "\n- User: 'Hello there!' -> Route: 'end', reply='Hello! How can I assist you today?'"
    )


    messages = [
        ("system", system_prompt),
        ("user", query)
    ]

    result: RouteDecision = router_llm.invoke(messages)

    initial_router_decision = result.route
    router_override_reason = None

    #Override router decision if web search is disabled and LLM chose 'web'
    if not web_search_enabled and result.route == "web":
        result.route = "rag"
        router_override_reason = "Web search disabled by user;redirected to RAG"

        print("Router decision is overriden; changed from 'web' to 'rag' because web search is disabled")

    print(f"Router final decision: {result.route}, Reply (if 'end'): {result.reply}")

    out  = {
        "messages" : state["messages"],
        "route" : result.route,
        "web_search_enabled" : web_search_enabled
    }

    if router_override_reason:
        out["initial_router_decision"] = initial_router_decision
        out["router_override_reason"] = router_override_reason

    if result.route == "end":
        out["messages"] = state["messages"] + [AIMessage(content=result.reply or "Hello!")]

    print("----- Existing router node -----")
    return out