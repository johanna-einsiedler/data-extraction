"""Live probe (2026-10-02): does each roster model accept temperature=0, and is T=0
deterministic? Three identical calls per configuration; prints the number of distinct
completions (1 = deterministic). Results recorded in analysis/tables/hpo/temperature_probe.md.
Zero-cost-ish: ~40 short calls."""
import os, json
from dotenv import load_dotenv; load_dotenv(".env")
from openai import OpenAI
P="In exactly two sentences, name two distinct risks of overfitting in clinical prediction models and one mitigation. Be specific."
C={"together":OpenAI(api_key=os.environ["TOGETHER_API_KEY"],base_url="https://api.together.xyz/v1",timeout=180),
   "openrouter":OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],base_url="https://openrouter.ai/api/v1",timeout=180),
   "openai":OpenAI(api_key=os.environ["OPENAI_API_KEY"],timeout=180)}
def trio(api,m,**kw):
    key="max_completion_tokens" if api=="openai" and m.startswith(("gpt-5","o")) else "max_tokens"
    outs=[(C[api].chat.completions.create(model=m,messages=[{"role":"user","content":P}],temperature=0,**{key:4000},**kw).choices[0].message.content or "").strip() for _ in range(3)]
    return len(set(outs))
for api,m,kw,label in (("together","zai-org/GLM-5.2",{"extra_body":{"reasoning":{"enabled":False}}},"open champion config (reasoning off)"),
                       ("openrouter","google/gemma-4-31B-it",{"extra_body":{"provider":{"require_parameters":True}}},"consumer champion config"),
                       ("openai","gpt-5.2",{},"temperature now accepted")):
    print(f"{m:28} {label:40} distinct/3 = {trio(api,m,**kw)}")
from google import genai; from google.genai import types
g=genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
outs=[g.models.generate_content(model="gemini-3.1-pro-preview",contents=P,config=types.GenerateContentConfig(temperature=0,max_output_tokens=4000)).text.strip() for _ in range(3)]
print(f"{'gemini-3.1-pro-preview':28} {'proprietary champion config':40} distinct/3 = {len(set(outs))}")
