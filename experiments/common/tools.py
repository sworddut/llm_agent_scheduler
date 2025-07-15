import arxiv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def arxiv_search(query: str, max_results: int = 5) -> str:
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
        search = arxiv.Search(
            query=query,
            max_results=max_results,
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

if __name__ == '__main__':
    # Example usage of the tool
    search_query = "Large Language Models in Software Engineering"
    papers = arxiv_search(search_query, max_results=2)
    print("--- Search Results ---")
    print(papers)
