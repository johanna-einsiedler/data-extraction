"""Probe: does z-ai/glm-5.2 via OpenRouter return answer-token logprobs on the frozen
P_open full-document batched base prompt (paper 103, tuning set) with reasoning disabled?
Usage: python probe_openrouter_glm.py <tag> <reasoning_enabled: on|off> [provider_order_csv]
Writes <tag>.raw.json and <tag>.summary.json into the script's directory. Never prints the key."""
import sys, os, json, time, pathlib
import requests
REPO = pathlib.Path('/Users/einsie0004/Documents/research/23_paper_extraction/pipeline_june_26')
sys.path.insert(0, str(REPO))
from src._data_loading import load_queries, build_system_prompt, load_full_document_text
from src import batched
from src.answer_generation.answer_generator import escape_braces_keep_context

tag, reasoning = sys.argv[1], sys.argv[2]
order = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] else None
api = sys.argv[4] if len(sys.argv) > 4 else 'openrouter'
OUT = pathlib.Path(__file__).parent

QUERIES = load_queries()
QUERIES = {q: v for q, v in QUERIES.items() if q != '4.1.1'}
prompt = batched.build_batched_prompt(QUERIES, 'base_prompt')
doc = load_full_document_text(REPO / 'data/parsed/MinerUParser/103_Dinga_2018.md')
sp = {}
for k, f in (('single_outcome', 'system_prompt_single_outcome.txt'), ('multiple_outcomes', 'system_prompt_multiple_outcomes.txt')):
    sp[k] = (REPO / 'query_creation' / f).read_text(encoding='utf-8')
system_prompt = build_system_prompt('103', fallback_single_prompt=sp['single_outcome'], fallback_multi_prompt=sp['multiple_outcomes'])
user = escape_braces_keep_context(prompt, 'context').format(context=doc)
messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user}]

provider = {'require_parameters': True, 'allow_fallbacks': order is None}
if order: provider['order'] = order
body = {
    'model': 'z-ai/glm-5.2', 'messages': messages, 'temperature': 0, 'max_tokens': 32768,
    'logprobs': True, 'top_logprobs': 5,
    'provider': provider, 'usage': {'include': True},
}
if reasoning == 'off':
    body['reasoning'] = {'enabled': False}
if api == 'together':
    body.pop('provider'); body.pop('usage'); body.pop('top_logprobs'); body['model'] = 'zai-org/GLM-5.2'
    headers = {'Authorization': f"Bearer {os.environ['TOGETHER_API_KEY']}", 'Content-Type': 'application/json'}
    url = 'https://api.together.xyz/v1/chat/completions'
else:
    headers = {'Authorization': f"Bearer {os.environ['OPENROUTER_API_KEY']}", 'Content-Type': 'application/json',
               'HTTP-Referer': 'https://localhost/research-probe', 'X-Title': 'logprob-probe'}
    url = 'https://openrouter.ai/api/v1/chat/completions'
summ_api = api
t0 = time.time()
r = requests.post(url, headers=headers, json=body, timeout=900)
dt = time.time() - t0
raw = r.json()
(OUT / f'{tag}.raw.json').write_text(json.dumps(raw, indent=1))
summ = {'tag': tag, 'api': summ_api, 'http': r.status_code, 'seconds': round(dt, 1), 'reasoning_requested': reasoning, 'order': order,
        'n_items_in_prompt': len(QUERIES), 'prompt_chars': len(user)}
if 'error' in raw:
    summ['error'] = raw['error']
else:
    ch = raw['choices'][0]
    summ['provider'] = raw.get('provider'); summ['finish_reason'] = ch.get('finish_reason')
    summ['usage'] = raw.get('usage')
    msg = ch.get('message', {})
    content = msg.get('content') or ''
    summ['content_chars'] = len(content); summ['reasoning_field_chars'] = len(msg.get('reasoning') or '')
    summ['reasoning_details'] = bool(msg.get('reasoning_details'))
    answers, others = batched.parse_batched_response(content, QUERIES)
    summ['parsed_answers'] = len([a for a in answers.values() if a not in (None, '')]) if isinstance(answers, dict) else None
    summ['answers_sample'] = {k: answers.get(k) for k in ('1.8', '1.9', '0.7', '2.1') if isinstance(answers, dict)}
    lp = (ch.get('logprobs') or {}).get('content')
    if lp:
        summ['logprob_tokens'] = len(lp)
        summ['tokens_with_top5'] = sum(1 for t in lp if len(t.get('top_logprobs') or []) >= 2)
        summ['first_tokens'] = [(t['token'], round(t['logprob'], 3), [(a['token'], round(a['logprob'], 2)) for a in (t.get('top_logprobs') or [])[:3]]) for t in lp[:12]]
        # locate the token after the "1.8" key in the content stream
        toks = [t['token'] for t in lp]; joined = ''.join(toks)
        i = joined.find('"1.8"')
        if i >= 0:
            pos = 0
            for j, tk in enumerate(toks):
                pos += len(tk)
                if pos > i + 5: break
            summ['tokens_after_1.8'] = [(t['token'], round(t['logprob'], 3), [(a['token'], round(a['logprob'], 2)) for a in (t.get('top_logprobs') or [])[:5]]) for t in lp[j:j+5]]
    elif isinstance(ch.get('logprobs'), dict) and ch['logprobs'].get('tokens'):
        tk = ch['logprobs']['tokens']; lps = ch['logprobs'].get('token_logprobs') or []
        summ['logprob_tokens'] = len(tk); summ['tokens_with_top5'] = 0
        summ['first_tokens'] = list(zip(tk[:12], [round(x,3) if x is not None else None for x in lps[:12]]))
    else:
        summ['logprob_tokens'] = 0; summ['logprobs_field'] = str(ch.get('logprobs'))[:300]
(OUT / f'{tag}.summary.json').write_text(json.dumps(summ, indent=1, default=str))
print(json.dumps(summ, indent=1, default=str)[:3500])
