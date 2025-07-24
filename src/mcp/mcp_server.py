from fastmcp import FastMCP
import arxiv
from typing import Union

mcp = FastMCP('local-arxiv-server')

@mcp.tool
def arxiv_search(query: str, max_results: Union[int, str] = 5) -> str:
    """Searches for papers on arXiv. Useful for academic research."""
    try:
        max_results_int = int(max_results)
    except (ValueError, TypeError):
        max_results_int = 5
    try:
        search = arxiv.Search(query=query, max_results=max_results_int, sort_by=arxiv.SortCriterion.Relevance)
        results = [f"Title: {r.title}\nAuthors: {', '.join(a.name for a in r.authors)}\nPublished: {r.published.strftime('%Y-%m-%d')}\nSummary: {r.summary.replace('n', ' ')}\nURL: {r.entry_id}" for r in search.results()]
        return "\n---\n".join(results) if results else "No papers found."
    except Exception as e:
        return f"Error during arXiv search: {e}"



if __name__ == "__main__":
    mcp.run()