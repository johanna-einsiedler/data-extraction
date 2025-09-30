# answer_generator.py
from typing import Any, Dict, List, Optional

from openai import OpenAI
from together import Together


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

        prompt = query.format(context=context_text)
        result: Dict[str, Any] = {}

        if self.api_type == "openai":
            if return_logprobs:
                # completions API doesn't support system prompts, so prepend manually
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                else:
                    full_prompt = prompt

                response = self.client.completions.create(
                    model=self.model,
                    prompt=full_prompt,
                    max_tokens=max_tokens,
                    logprobs=0,  # returns logprobs for all tokens
                )
                choice = response.choices[0]
                result["text"] = choice.text
                result["logprobs"] = {
                    "tokens": choice.logprobs.tokens,
                    "token_logprobs": choice.logprobs.token_logprobs,
                }
            else:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                result["text"] = response.choices[0].message.content

        elif self.api_type == "together":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_output_tokens=max_tokens,
            )
            result["text"] = response.choices[0].message.content
            # Together API does not expose token logprobs directly
            result["logprobs"] = None

        return result
