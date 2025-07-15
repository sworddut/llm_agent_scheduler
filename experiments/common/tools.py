import arxiv
import logging
from typing import Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def arxiv_search(query: str, max_results: Union[int, str] = 5) -> str:
    """
    Searches for papers on arXiv based on a query and returns the top results.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return.

    Returns:
        str: A formatted string containing the search results, including title, authors, and summary for each paper.
    """
    try:
        logger.info(f"Executing arxiv_search with query: '{query}' and max_results: {max_results}")
                # Ensure max_results is an integer
        try:
            max_results_int = int(max_results)
        except (ValueError, TypeError):
            max_results_int = 5 # Default to 5 if conversion fails

        search = arxiv.Search(
            query=query,
            max_results=max_results_int,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for r in search.results():
            results.append(
                f"Title: {r.title}\n"
                f"Authors: {', '.join(author.name for author in r.authors)}\n"
                f"Published: {r.published.strftime('%Y-%m-%d')}\n"
                f"Summary: {r.summary.replace('n', ' ')}\n"
                f"URL: {r.entry_id}"
            )
        
        if not results:
            return "No papers found for the given query."

        return "\n---\n".join(results)

    except Exception as e:
        logger.error(f"An error occurred during arXiv search: {e}")
        return f"Error: Could not perform search. {str(e)}"

# Tool definition for the scheduler and planner
arxiv_search_tool = {
    "type": "function",
    "function": {
        "name": "arxiv_search",
        "description": "Searches for papers on arXiv based on a query and returns the top results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for papers on arXiv."
                },
                "max_results": {
                    "type": "integer",
                    "description": "The maximum number of papers to return."
                }
            },
            "required": ["query"]
        }
    },
    "callable": arxiv_search
}

if __name__ == '__main__':
    # Example usage of the tool
    search_query = "Large Language Models in Software Engineering"
    papers = arxiv_search(search_query, max_results=2)
    print("--- Search Results ---")
    print(papers)
