# Submission — HW Agentic RAG

**Name:** Prajwal Rudresh
**Student ID:** PXR240019

## LangSmith trace URL
https://smith.langchain.com/o/cf5202e1-4803-4c98-a934-63ef5bdd97dd/projects/p/3bbf912e-6dce-4330-8be4-518aa64fd610

## Final answer text
### Directors of Tesla and Their LinkedIn Handles:
1. **Elon Musk** - [LinkedIn](https://www.linkedin.com/in/ceomrmusk-96ba32324)
2. **Robyn Denholm** - [LinkedIn](https://au.linkedin.com/in/robyn-denholm-a807795)
3. **Ira Ehrenpreis** - [LinkedIn](https://www.linkedin.com/in/iraehrenpreis)
4. **Joseph Gebbia** - [LinkedIn](https://www.linkedin.com/in/jgebbia)
5. **James Murdoch** - [LinkedIn](https://www.linkedin.com/in/jamesrmurdoch)
6. **Kimbal Musk** - [LinkedIn](https://www.linkedin.com/in/kimbalmusk)
7. **JB Straubel** - [LinkedIn](https://www.linkedin.com/in/jb-straubel-b694981)
8. **Kathleen Wilson-Thompson** - [LinkedIn](https://www.linkedin.com/in/kathleen-wilson-thompson-275654201)

### Financial Goals of Tesla for 2023:
Tesla's financial goals for 2023 include meeting capital expenditure requirements and
generating significant tax revenues. Specifically, Tesla aimed to spend RMB 14.08 billion
in capital expenditures by the end of 2023 and generate RMB 2.23 billion in annual tax
revenues. These goals have been achieved, and Tesla expects to continue meeting these
financial targets based on current spending and sales levels.

### Next Auto Show Participation:
Tesla will participate in the 2023 Detroit Auto Show, which will take place from September
13 to 24 at Huntington Place in Detroit. Tesla will showcase its vehicles at the "Powering
Michigan EV Experience" and offer visitors test drives. Additionally, Tesla will also
participate in the 2023 Munich Auto Show.

## What the agent did (1 paragraph)
The agent decomposed the demo question into four parts and selected a different tool for
each. For the directors and LinkedIn handles, it called `company_directors_information`,
which used gpt-4o-mini to extract 8 director names from the cached 10-K end-section, then
looked up each LinkedIn URL via SerpAPI. For Tesla's financial goals, it called
`vector_reranker_search`, performing a FAISS semantic search over the 10-K chunks followed
by Flashrank reranking to surface the 5 most relevant passages. For the next auto show, it
called `web_search` via Tavily, correctly identifying this as a post-cutoff question
requiring live web data. The question asks about financial goals "this year," but the most
recent 10-K in the vector store is from 2023, so the agent answered with 2023 data. This
is reasonable behavior — the agent uses the most recent available data. To make it more
predictable, the prompt could explicitly state the filing year (e.g., "based on the 2023
10-K filing") so the agent does not interpret "this year" ambiguously.

## Notes
The two-web-search-API design (Tavily for general queries, SerpAPI for LinkedIn lookups)
was a key insight — Tavily's reranking would interfere with the precision of the
`site:linkedin.com/in/` operator that SerpAPI passes through verbatim. The LinkedIn cache
(`_linkedin_cache`) was essential to avoid burning SerpAPI quota on repeated runs.