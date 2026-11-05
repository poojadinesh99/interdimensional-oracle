"""MCP server exposing the Oracle's hybrid retrieval as a tool.

Any MCP client (Claude Desktop, Claude Code, Cursor, an agent framework) can
call `search_rick_and_morty` to get ranked passages from the local knowledge
base — the same FTS5 + embeddings + RRF retrieval the /api/chat endpoint uses —
and do its own reasoning over them.

Run over stdio (what desktop MCP clients expect):
    python -m backend.app.mcp_server

Run over Streamable HTTP (served at http://localhost:8001/mcp):
    python -m backend.app.mcp_server --http --port 8001
"""

from __future__ import annotations

import argparse
from typing import List

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .guardrails import validate_question
from . import retrieval

MAX_TOP_K = 10
MAX_PASSAGE_CHARS = 2000

mcp = MCPServer(
	name="interdimensional-oracle",
	instructions=(
		"Knowledge base of Rick & Morty characters, locations and episodes "
		"(from the Rick and Morty API). Call search_rick_and_morty to retrieve "
		"source passages, and answer only from what it returns."
	),
)


class Passage(BaseModel):
	id: str = Field(description="Document id, e.g. 'character:1' or 'episode:28'")
	title: str
	source: str = Field(description="Rick and Morty API URL the document came from")
	score: float = Field(description="Fused relevance score, normalised to 0..1 (1 = best match)")
	content: str


class SearchResult(BaseModel):
	query: str
	passages: List[Passage]


@mcp.tool(
	title="Search the Rick & Morty knowledge base",
	annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
)
def search_rick_and_morty(query: str, top_k: int = 3) -> SearchResult:
	"""Hybrid search (SQLite FTS5 keyword match + sentence-transformer embeddings,
	merged with Reciprocal Rank Fusion) over Rick & Morty characters, locations
	and episodes. Returns the top_k most relevant passages (1-10) with their source URLs.
	"""

	try:
		clean_query = validate_question(query)  # same guardrails as /api/chat
	except ValueError as exc:
		# Expected refusal: surface the reason to the client, not as a server crash.
		raise ToolError(str(exc)) from exc
	top_k = max(1, min(int(top_k), MAX_TOP_K))
	docs = retrieval.retrieve(clean_query, top_k=top_k)
	return SearchResult(
		query=clean_query,
		passages=[
			Passage(
				id=str(doc["id"]),
				title=str(doc["title"]),
				source=str(doc.get("source") or ""),
				score=round(float(doc["score"]), 4),
				content=str(doc["content"])[:MAX_PASSAGE_CHARS],
			)
			for doc in docs
		],
	)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8001)
	args = parser.parse_args()

	if args.http:
		mcp.run(transport="streamable-http", host=args.host, port=args.port)
	else:
		mcp.run(transport="stdio")


if __name__ == "__main__":
	main()
