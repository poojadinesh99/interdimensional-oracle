MAX_QUESTION_LENGTH = 4000
PROMPT_INJECTION_PHRASES = (
	"ignore previous instructions",
	"system prompt",
	"developer instructions",
	"forget your instructions",
)
OFF_TOPIC_KEYWORDS = (
	"python",
	"java",
	"javascript",
	"politics",
	"president",
	"election",
	"medical",
	"doctor",
	"disease",
	"lawyer",
	"legal",
	"recipe",
	"weather",
)


def validate_question(question: str) -> str:
	"""Validate a user question before retrieval."""

	clean_question = question.strip()
	if not clean_question:
		raise ValueError("Please enter a question.")
	if len(clean_question) > MAX_QUESTION_LENGTH:
		raise ValueError("Your question is too long. Please keep it under 4000 characters.")
	if contains_prompt_injection(clean_question):
		raise ValueError("Prompt injection attempts are not allowed.")
	if is_off_topic(clean_question):
		raise ValueError("I can only answer questions about the Rick & Morty universe.")
	return clean_question


def contains_prompt_injection(text: str) -> bool:
	"""Detect common prompt injection attempts."""

	lowered = text.lower()
	return any(phrase in lowered for phrase in PROMPT_INJECTION_PHRASES)


def is_off_topic(text: str) -> bool:
	"""Detect obvious requests outside the Rick & Morty knowledge base."""

	lowered = text.lower()
	return any(keyword in lowered for keyword in OFF_TOPIC_KEYWORDS)
