"""
Wire everything together: ingest filings, build the retriever, build the tools,
build the ReAct agent, and run it.

The notebook used `langchain_classic.agents.create_react_agent + AgentExecutor`.
We use `langgraph.prebuilt.create_react_agent` instead — it's the LangGraph
equivalent and returns a compiled state graph (just like the ones we built
by hand in class). Same ReAct pattern, cleaner integration with LangSmith.
PRE-FILLED FOR YOU — the import + invocation pattern is non-obvious; we want
you to study it, not rediscover it.
"""

import os
from dotenv import load_dotenv

# Load env BEFORE any imports that might construct LLM clients.
load_dotenv()

# Required keys: fail fast if any are missing. LangSmith is REQUIRED for HW.
_REQUIRED = ["OPENAI_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY", "LANGCHAIN_API_KEY"]
for k in _REQUIRED:
    if not os.getenv(k):
        raise ValueError(f"{k} is required for this assignment. See env.template.")

# LangSmith tracing is REQUIRED for HW.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "hw_agentic_rag_spring_2026")

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# LangGraph's prebuilt ReAct agent. This is the LangGraph version of the
# by-hand graphs you built in Reflection/Reflexion — same loop pattern,
# but pre-baked so you don't have to wire it node by node.
from langgraph.prebuilt import create_react_agent

from config import CONFIG
from prompts import DEMO_QUESTION
from ingestion import load_and_process_filings
from retriever import create_vector_store, create_retriever_with_reranking
from tools import VectorRerankerSearchTool, WebSearchTool, CompanyDirectorsTool


def build_pipeline():
    """
    TODO M6.1: Build the data pipeline — ingest, embed, wrap with reranker.

    Chain together the three functions you wrote in M3 and M4
    (load_and_process_filings, create_vector_store,
    create_retriever_with_reranking) using the relevant CONFIG keys.
    Return the retriever AND the director_sections (the directors tool
    needs the latter). See MILESTONES.md M6.
    """

    chunks, director_sections = load_and_process_filings(
        urls=CONFIG["companyFilingUrls"],
        chunk_size=CONFIG["chunkSize"],
        chunk_overlap=CONFIG["chunkOverlap"],
        user_agent=CONFIG["userAgentHeader"],
    )
    vectorstore = create_vector_store(chunks, CONFIG["embeddingModelName"])
    retriever = create_retriever_with_reranking(
        vectorstore=vectorstore,
        num_retrieved=CONFIG["numRetrievedDocuments"],
        reranker_model=CONFIG["rerankerModel"],
        num_reranked=CONFIG["numRerankedDocuments"],
    )
    return retriever, director_sections

    raise NotImplementedError("Build build_pipeline — see Milestone 6")


def build_tools(retriever, director_sections):
    """
    TODO M6.2: Construct the three BaseTool instances and return them as a list.

    Each tool's constructor takes either the retriever or the
    director_sections, plus a num_results / extraction config from CONFIG.
    The agent picks among these tools at runtime. See MILESTONES.md M6.
    """
    return [
        VectorRerankerSearchTool(retriever=retriever, num_results=CONFIG["numRetrieverToolResults"]),
        WebSearchTool(num_results=CONFIG["numWebToolResults"]),
        CompanyDirectorsTool(
            director_sections=director_sections,
            extraction_model=CONFIG["nameExtractionModel"],
            extraction_temperature=CONFIG["nameExtractionModelTemperature"],
        ),
    ]



    raise NotImplementedError("Build build_tools — see Milestone 6")


def build_agent(tools):
    """
    TODO M6.3: Build the LangGraph ReAct agent.

    Two real lines: instantiate ChatOpenAI with CONFIG's react model + temp,
    then hand it (with the tools) to `create_react_agent` — already imported
    at the top of the file from langgraph.prebuilt. See MILESTONES.md M6.
    """
    llm = ChatOpenAI(
        model=CONFIG["reactModelName"],
        temperature=CONFIG["reactModelTemperature"],
    )
    return create_react_agent(llm, tools)


    raise NotImplementedError("Build build_agent — see Milestone 6")


# ============================================================================
# TEST FUNCTIONS — run individual pieces without invoking the full agent.
# Each one is independent. Useful when debugging to avoid burning API quota.
# ============================================================================

def test_ingestion():
    print("=" * 60)
    print("Testing ingestion...")
    print("=" * 60)
    chunks, director_sections = load_and_process_filings(
        urls=CONFIG["companyFilingUrls"],
        chunk_size=CONFIG["chunkSize"],
        chunk_overlap=CONFIG["chunkOverlap"],
        user_agent=CONFIG["userAgentHeader"],
    )
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Companies with director sections: {list(director_sections.keys())}")


def test_retriever():
    print("=" * 60)
    print("Testing retriever pipeline...")
    print("=" * 60)
    retriever, _ = build_pipeline()
    docs = retriever.invoke("What are Tesla's financial goals?")
    print(f"\nReturned {len(docs)} reranked docs.")
    for d in docs[:2]:
        print(f"\n[{d.metadata.get('company')}] {d.page_content[:200]}...")


def test_tools_independently():
    print("=" * 60)
    print("Testing each tool independently...")
    print("=" * 60)
    retriever, director_sections = build_pipeline()
    tools = build_tools(retriever, director_sections)

    print("\n--- VectorRerankerSearchTool ---")
    print(tools[0]._run("What are Tesla's financial goals for 2024?")[:400])

    print("\n--- WebSearchTool (Tavily) ---")
    print(tools[1]._run("Tesla auto show 2026")[:400])

    print("\n--- CompanyDirectorsTool (SerpAPI for LinkedIn) ---")
    print(tools[2]._run("Tesla, true")[:1500])


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Set TEST_TO_RUN to one of: "ingestion" / "retriever" / "tools" / None
    # None = run the full agent end-to-end with the demo question.
    TEST_TO_RUN = "None"

    tests = {
        "ingestion": test_ingestion,
        "retriever": test_retriever,
        "tools": test_tools_independently,
    }

    if TEST_TO_RUN is not None and TEST_TO_RUN in tests:
        tests[TEST_TO_RUN]()
    else:
        # TODO M7: Wire the full agent run end-to-end.
        #
        # Build pipeline, tools, and agent (the helpers you wrote in M6).
        # Save the graph picture (the agent has a `.get_graph()` method;
        # draw_mermaid_png writes a PNG you'll submit). Invoke the agent
        # with the demo question — same `{"messages": [HumanMessage(...)]}`
        # signature you used in Reflection. Extract the final AIMessage
        # from result["messages"] and print it under a banner.
        #
        # Don't forget to print a hint at the end pointing students at
        # https://smith.langchain.com so they can find their trace.
        #
        # See MILESTONES.md M7.
        print(f"\n{'='*60}")
        print(f"Running full agent — LangSmith project: {os.getenv('LANGCHAIN_PROJECT')}")
        print(f"{'='*60}\n")

        retriever, director_sections = build_pipeline()
        tools = build_tools(retriever, director_sections)
        agent = build_agent(tools)

        agent.get_graph().draw_mermaid_png(output_file_path="agent_graph.png")
        print("Graph saved to agent_graph.png\n")

        print(f"DEMO QUESTION:\n{DEMO_QUESTION}\n")

        result = agent.invoke(
            {"messages": [HumanMessage(content=DEMO_QUESTION)]},
            config={"recursion_limit": 25},
        )

        final_answer = [m for m in result["messages"] if isinstance(m, AIMessage)][-1].content

        print(f"\n{'='*60}")
        print("FINAL ANSWER")
        print(f"{'='*60}")
        print(final_answer)
        print(f"\n{'='*60}")
        print("View your trace at: https://smith.langchain.com")
        print(f"{'='*60}")


        # raise NotImplementedError("Build the full agent run — see Milestone 7")
