import time
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain, extract_text, invoke_with_retry

def run_research_pipeline(topic : str) -> dict:

    state = {}

    def on_retry(msg):
        print(f"\n[Warning] {msg}")

    # search agent working 
    print("\n" + " =" * 50)
    print("step 1 - search agent is working ...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = invoke_with_retry(search_agent, {
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    }, on_retry=on_retry)
    state["search_results"] = extract_text(search_result['messages'][-1].content)

    print("\n search result ", state['search_results'])

    time.sleep(2)

    # step 2 - reader agent 
    print("\n" + " =" * 50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("=" * 50)

    reader_agent = build_reader_agent()
    reader_result = invoke_with_retry(reader_agent, {
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    }, on_retry=on_retry)

    state['scraped_content'] = extract_text(reader_result['messages'][-1].content)

    print("\nscraped content: \n", state['scraped_content'])

    time.sleep(2)

    # step 3 - writer chain 
    print("\n" + " =" * 50)
    print("step 3 - Writer is drafting the report ...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = extract_text(invoke_with_retry(writer_chain, {
        "topic": topic,
        "research": research_combined
    }, on_retry=on_retry))

    print("\n Final Report\n", state['report'])

    time.sleep(2)

    # step 4 - critic report 
    print("\n" + " =" * 50)
    print("step 4 - critic is reviewing the report ")
    print("=" * 50)

    state["feedback"] = extract_text(invoke_with_retry(critic_chain, {
        "report": state['report']
    }, on_retry=on_retry))

    print("\n critic report \n", state['feedback'])

    return state

if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)
