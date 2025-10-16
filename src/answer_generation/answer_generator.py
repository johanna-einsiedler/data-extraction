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
        self.model = model

        if self.api_type == "openai":
            self.client = OpenAI(api_key=api_key)

        elif self.api_type == "together":
            self.client = Together(api_key=api_key)
        else:
            raise ValueError(f"Unsupported api_type: {api_type}")

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

        if self.api_type == "openai":
            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=messages,
            #     top_logprobs=20,
            #     # max_tokens=max_tokens,
            # )
            if self.model == "o3-mini-2025-01-31":
                response = self.client.responses.create(
                    instructions=system_prompt,
                    model=self.model,
                    input=prompt,  # top_logprobs=20
                )
                text = response.output_text
                token_logprobs = np.nan

            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # not wrapped again!
                    logprobs=True,
                    # top_logprobs=5,
                )
                text = response.choices[0].message.content
                tokens_list = response.choices[0].logprobs.content

                # Build dict: token -> logprob
                token_logprobs = {t.token: t.logprob for t in tokens_list}
                print(token_logprobs)

            result["text"] = text
            result["logprobs"] = token_logprobs

        elif self.api_type == "together":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                logprobs=1,
                # max_output_tokens=max_tokens,
            )
            result["text"] = response.choices[0].message.content
            # Together API does not expose token logprobs directly
            result["logprobs"] = response.choices[0].logprobs

        return result
