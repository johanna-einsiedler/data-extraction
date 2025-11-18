# answer_generator.py
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI
from together import Together


def escape_braces_keep_context(template: str, placeholder: str = "context") -> str:
    """
    Escape all braces in the template except the one for {context}.
    """
    # Temporarily replace the placeholder with a token
    token = "___PLACEHOLDER___"
    template = template.replace(f"{{{placeholder}}}", token)

    # Escape all remaining braces
    template = template.replace("{", "{{").replace("}", "}}")

    # Put the placeholder back
    template = template.replace(token, f"{{{placeholder}}}")
    return template


class AnswerGenerator:
    def __init__(
        self, api_type: str = "openai", api_key: str = None, model: str = None
    ):
        """
        api_type: "openai" or "together"
        api_key: API key for OpenAI or Together
        model: model name, e.g., "gpt-4.1" or "o3-mini-high"
        """
        self.api_type = api_type.lower()
        self.api_key = api_key
        self.model = model
        self._client = None

        if self.api_type not in {"openai", "together"}:
            raise ValueError(f"Unsupported api_type: {api_type}")

    def _ensure_client(self):
        if not self.api_key:
            raise ValueError(
                f"API key is required for {self.api_type.title()} AnswerGenerator."
            )
        if self._client is None:
            if self.api_type == "openai":
                self._client = OpenAI(api_key=self.api_key)
            else:
                self._client = Together(api_key=self.api_key)
        return self._client

    def generate(
        self,
        query: str,
        chunks: List[str],
        # prompt_template: Optional[str] = None,
        max_tokens: int = 1024,
        return_logprobs: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer using retrieved chunks and a prompt.
        Returns a dict with:
            - 'text': generated answer
            - 'logprobs': optional dict with 'tokens' and 'token_logprobs'
        """
        context_text = "\n".join([str(chunk[0]) for chunk in chunks])

        # if prompt_template is None:
        #     prompt_template = (
        #         "Use the following context to answer the question.\n\n"
        #         "Context:\n{context}\n\n"
        #         "Question: {query}\n\n"
        #         "Answer concisely:"
        #     )
        query = escape_braces_keep_context(query, "context")
        prompt = query.format(context=context_text)
        result: Dict[str, Any] = {}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = self._ensure_client()

        if self.api_type == "openai":
            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=messages,
            #     top_logprobs=20,
            #     # max_tokens=max_tokens,
            # )
            if self.model == "o3-mini-2025-01-31":
                response = client.responses.create(
                    instructions=system_prompt,
                    model=self.model,
                    input=prompt,  # top_logprobs=20
                )
                text = response.output_text
                token_logprobs = np.nan

            else:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # not wrapped again!
                    logprobs=True,
                    # top_logprobs=5,
                )
                text = response.choices[0].message.content
                tokens_list = response.choices[0].logprobs.content

                # Build dict: token -> logprob
                token_logprobs = {t.token: t.logprob for t in tokens_list}

            result["text"] = text
            result["logprobs"] = token_logprobs

        elif self.api_type == "together":
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                logprobs=1,
                # max_output_tokens=max_tokens,
            )
            result["text"] = response.choices[0].message.content
            logprob_payload = getattr(response.choices[0], "logprobs", None)
            result["logprobs"] = logprob_payload if logprob_payload is not None else None

        return result
