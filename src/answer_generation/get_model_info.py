from dotenv import find_dotenv, load_dotenv
from together import Together

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
import json
import os

from answer_generation.answer_generator import AnswerGenerator

client = Together()

models = client.models.list()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
together_model_list = [
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "openai/gpt-oss-20b",
    "moonshotai/Kimi-K2-Instruct",
    "zai-org/GLM-4.5-Air-FP8",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "google/gemma-2-27b-it",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "deepcogito/cogito-v2-preview-llama-405B",
    "Qwen/QwQ-32B",
    "lgai/exaone-deep-32b",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    "arcee-ai/AFM-4.5B",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "arcee-ai/maestro-reasoning",
    "openai/gpt-oss-120b",
]

filtered_models = [m for m in models if m.id in together_model_list]


def model_to_dict(model):
    """Convert your ModelObject to a serializable dict."""
    return {
        "id": model.id,
        "display_name": model.display_name,
        "organization": model.organization,
        "context_length": model.context_length,
        "pricing": {
            "input": model.pricing.input,
            "output": model.pricing.output,
        },
        "link": model.link,
        "api_type": "together",
    }


filtered_data = [model_to_dict(m) for m in filtered_models]


filtered_data.append(
    {
        "id": "o3-mini-2025-01-31",
        "display_name": "O3 Mini",
        "organization": "OpenAI",
        "context_length": 100000,
        "pricing": {"input": 1.1, "output": 4.4},
        "link": "https://platform.openai.com/docs/models/o3-mini",
        "api_type": "openai",
    }
)


filtered_data.append(
    {
        "id": "gpt-4.1-2025-04-14",
        "display_name": "GPT-4.1",
        "organization": "OpenAI",
        "context_length": 1047576,
        "pricing": {"input": 2, "output": 8},
        "link": "https://platform.openai.com/docs/models/gpt-4.1",
        "api_type": "openai",
    }
)
# Save to JSON
with open("filtered_models.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)
