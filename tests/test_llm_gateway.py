import json
from dataclasses import replace

import httpx
import pytest

from backend.app.config import settings
from backend.app.llm_clients import LiteLLMGatewayClient, LLMGatewayError
from backend.app.rag_pipeline import build_llm_client


def _client(handler) -> LiteLLMGatewayClient:
	http = httpx.Client(
		base_url="http://gateway.test",
		headers={"Authorization": "Bearer sk-test"},
		transport=httpx.MockTransport(handler),
	)
	return LiteLLMGatewayClient(base_url="http://gateway.test", model="oracle-default", http_client=http)


def test_sends_openai_format_request_with_alias_and_key():
	seen = {}

	def handler(request: httpx.Request) -> httpx.Response:
		seen["path"] = request.url.path
		seen["auth"] = request.headers.get("authorization")
		seen["body"] = json.loads(request.content)
		return httpx.Response(200, json={"choices": [{"message": {"content": "  Rick is a scientist.  "}}]})

	answer = _client(handler).generate("Who is Rick?")

	assert answer == "Rick is a scientist."
	assert seen["path"] == "/chat/completions"
	assert seen["auth"] == "Bearer sk-test"
	assert seen["body"]["model"] == "oracle-default"
	assert seen["body"]["messages"] == [{"role": "user", "content": "Who is Rick?"}]


def test_gateway_error_status_raises():
	client = _client(lambda r: httpx.Response(429, json={"error": "rate limited"}))
	with pytest.raises(LLMGatewayError, match="429"):
		client.generate("hi")


def test_unexpected_response_shape_raises():
	client = _client(lambda r: httpx.Response(200, json={"nope": True}))
	with pytest.raises(LLMGatewayError, match="unexpected response"):
		client.generate("hi")


def test_unreachable_gateway_raises():
	def handler(request):
		raise httpx.ConnectError("connection refused", request=request)

	with pytest.raises(LLMGatewayError, match="unreachable"):
		_client(handler).generate("hi")


def test_provider_selection():
	assert isinstance(build_llm_client(replace(settings, llm_provider="litellm")), LiteLLMGatewayClient)
	with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
		build_llm_client(replace(settings, llm_provider="nope"))
