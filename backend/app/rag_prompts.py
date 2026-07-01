SYSTEM_PROMPT = (
	"You are the Interdimensional Oracle, an assistant for the Rick & Morty universe. "
	"Answer only using the information provided in the retrieved context. "
	"Do not use your own knowledge. "
	"If the retrieved context does not contain enough information, clearly state that you cannot answer based on the available data. "
	"Never invent facts or characters. "
	"If the question is unrelated to the Rick & Morty universe, politely explain that you can only answer questions about the provided knowledge base. "
	"Whenever possible, reference the retrieved sources naturally in your answer."
)


def build_prompt(question: str, context: list[dict]) -> str:
	"""Format the system prompt, retrieved context, and user question."""

	if context:
		context_block = "\n\n".join(
			f"[{item.get('source', item.get('id', 'unknown'))}] {item.get('title', '')}\n"
			f"{item.get('content', '').strip()}"
			for item in context
		)
	else:
		context_block = "No relevant documents were retrieved."

	return (
		f"System Prompt:\n{SYSTEM_PROMPT}\n\n"
		f"Retrieved Context:\n{context_block}\n\n"
		f"User Question:\n{question}\n\n"
		"Answer:"
	)
