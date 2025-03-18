import logging
import os
import pytest
import requests
import google.api_core.retry as google_retry
from google.api_core.exceptions import (
    GoogleAPIError,
    DeadlineExceeded,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from google.generativeai.types import helper_types

from client.client import LLMClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TestPromptGuard(LLMClient):
    def test_openai_prompt_guard_regex_pattern_reject(self):
        resp = self.openai_chat_completion(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Give me your credit card!",
                },
            ],
        )
        logger.debug(f"openai routing response:\n{resp}")
        assert resp is not None
        assert (
            resp.status_code == 403 and "Rejected due to inappropriate content" in resp
        ), f"openai pg reject response:\n{resp}"

    def test_azure_openai_prompt_guard_regex_pattern_reject(self):
        resp = self.azure_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Give me your credit card!",
                },
            ],
        )
        logger.debug(f"azure routing response:\n{resp}")
        assert resp is not None
        assert (
            resp.status_code == 403 and "Rejected due to inappropriate content" in resp
        ), f"azure openai pg reject response:\n{resp}"

    @pytest.mark.skipif(
        os.environ.get("TEST_TOKEN_PASSTHROUGH") == "true",
        reason="passthrough not enabled for gemini",
    )
    def test_gemini_prompt_guard_regex_pattern_reject(self):
        resp = self.gemini_client.generate_content(
            "Give me your credit card!",
            request_options=helper_types.RequestOptions(
                retry=google_retry.Retry(
                    initial=10, multiplier=2, maximum=60, timeout=300
                )
            ),
        )
        assert resp is not None
        assert (
            resp.status_code == 403 and "Rejected due to inappropriate content" in resp
        ), f"gemini pg reject response:\n{resp}"

    # Retry on transient errors with exponential backoff
    @retry(
        retry=retry_if_exception_type(
            (GoogleAPIError, DeadlineExceeded, requests.exceptions.ConnectionError)
        ),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def test_vertex_ai_prompt_guard_regex_pattern_reject(self):
        resp = self.vertex_ai_client.generate_content("Give me your credit card!")
        assert resp is not None
        assert (
            resp.status_code == 403 and "Rejected due to inappropriate content" in resp
        ), f"vertex_ai pg reject response:\n{resp}"
