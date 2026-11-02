"""Model access, kept separate for the two roles in this project.

generate_response(...)  dialogue generation: the user agent, the system agent and
                        the BFI probe of the trait prompts
judge_response(...)     evaluation: the LLM judges (User Trait Score, system quality,
                        pairwise satisfaction, system adaptation)

Both take the model and the provider explicitly; nothing is hard-coded here.

Providers ("api_source" in the generation config, "judge_provider" in the judge configs):
  openai      OpenAI API                 key: openai_api_key   (GPT-4o)
  open_route  OpenRouter                 key: routeai_api_key  (Gemini 2.0 Flash, GPT-5-mini judge)
  ollama      Ollama Cloud               key: OLLAMA_API_KEY   (Qwen3-Next-80B, Gemma-4-31B judge)

API keys are read from .env (see .env.example).
"""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_HOST = "https://ollama.com"
PROVIDERS = ("openai", "open_route", "ollama")


def _with_retries(call, max_retries=10, max_wait=128):
    """Run `call()` until it returns non-empty text, with exponential back-off."""
    wait = 2
    for attempt in range(max_retries):
        try:
            text = call()
            if not text:
                raise ValueError("empty response")
            return text
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Error after {max_retries} attempts: {e}")
            print(f"Attempt {attempt + 1} failed: {e}\nRetrying in {wait} seconds...")
            time.sleep(wait)
            wait = min(wait * 2, max_wait)


def _openai(prompt, model, options):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["openai_api_key"])
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], n=1, **options)
    return response.choices[0].message.content


def _open_route(prompt, model, options):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], **options}
    r = requests.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {os.environ['routeai_api_key']}"},
                      json=payload, timeout=300)
    return r.json()["choices"][0]["message"]["content"]


def _ollama(prompt, model, options):
    from ollama import Client
    client = Client(host=OLLAMA_HOST, headers={"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]})
    kwargs = {"options": options} if options else {}
    response = client.chat(model=model, messages=[{"role": "user", "content": prompt}], **kwargs)
    return response["message"]["content"]


_CALL = {"openai": _openai, "open_route": _open_route, "ollama": _ollama}


def _check(provider, model):
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}, got {provider!r}")
    if not model:
        raise ValueError("a model name is required")


def generate_response(prompt, model_name, api_source, max_tokens=250, temperature=0.7, top_p=1.0):
    """One turn of dialogue generation (user agent, system agent) or a BFI probe answer.

    Paper settings for all three models (GPT-4o, Gemini 2.0 Flash, Qwen3-Next-80B):
    temperature 0.7, top-p 1.0, at most 250 output tokens per call
    (gpt_temperature / gpt_max_tokens in the config).
    """
    _check(api_source, model_name)
    if api_source == "ollama":
        options = {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}
    else:
        options = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    return _with_retries(lambda: _CALL[api_source](prompt, model_name, options))


def judge_response(prompt, judge_model, judge_provider, max_tokens=4000, reasoning_effort=None):
    """One LLM-judge call.

    Paper judges:
      GPT-5-mini  judge_provider="open_route", judge_model="openai/gpt-5-mini",
                  reasoning_effort="low", max_tokens=4000 (covers reasoning + answer)
      Gemma       judge_provider="ollama", judge_model="gemma4:31b-cloud" (provider defaults)
    No sampling options are set, as in the paper's judge runs.
    """
    _check(judge_provider, judge_model)
    if judge_provider == "ollama":
        options = {}
    else:
        options = {"max_tokens": max_tokens}
        if reasoning_effort:
            if judge_provider == "open_route":
                options["reasoning"] = {"effort": reasoning_effort}
            else:
                options = {"max_completion_tokens": max_tokens, "reasoning_effort": reasoning_effort}
    return _with_retries(lambda: _CALL[judge_provider](prompt, judge_model, options), max_retries=8)
