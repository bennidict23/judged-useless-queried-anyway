"""vLLM inference with explicit, logged decoding modes and truncation status."""
from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace
from pathlib import Path

from config import TEMPERATURE, SEED, VLLM_TENSOR_PARALLEL, stable_seed, REPLICATE, SHUFFLE_SALT


def decoding_config(model_path, thinking=False, max_tokens=None):
    qwen3 = 'qwen3' in str(model_path).lower()
    if thinking and not qwen3:
        raise ValueError('--thinking is currently supported only for Qwen3')
    config = {
        'enable_thinking': thinking if qwen3 else None,
        'temperature': (0.6 if thinking else 0.7) if qwen3 else TEMPERATURE,
        'top_p': (0.95 if thinking else 0.8) if qwen3 else 1.0,
        'top_k': 20 if qwen3 else -1,
        'max_tokens': max_tokens if max_tokens is not None else (8192 if thinking else (1024 if qwen3 else 400)),
        # MAX_MODEL_LEN env override: thinking mode on a 24 GB card cannot reserve KV cache for 32k.
        'max_model_len': int(os.environ['MAX_MODEL_LEN']) if os.environ.get('MAX_MODEL_LEN') else (32768 if thinking else 8192),
        'seed': SEED,
        'seed_policy': 'prompt_sha256' if qwen3 else 'fixed',
    }
    if REPLICATE:   # addendum 10: new sampling seeds (and, in config, a new misleading-paragraph order)
        config.update(seed=stable_seed(SEED, 'replicate', REPLICATE) % (2**31), replicate=REPLICATE, shuffle_salt=SHUFFLE_SALT)
    # Sampling overrides for controls (e.g. run non-thinking with thinking-mode sampling): recorded in generation_config.
    if os.environ.get('TEMPERATURE_OVERRIDE'):
        config['temperature'] = float(os.environ['TEMPERATURE_OVERRIDE']); config['sampling_override'] = True
    if os.environ.get('TOP_P_OVERRIDE'):
        config['top_p'] = float(os.environ['TOP_P_OVERRIDE']); config['sampling_override'] = True
    if config['max_tokens'] <= 0 or config['max_tokens'] >= config['max_model_len']:
        raise ValueError('generation budget must be positive and smaller than context window')
    return config


class BatchLocalLLM:
    def __init__(self, model_path: str, gpu_ids: str = '0', thinking=False, max_tokens=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        from vllm import LLM
        self.generation_config = decoding_config(model_path, thinking, max_tokens)
        self.generation_config['model_path'] = model_path
        self.max_tokens = self.generation_config['max_tokens']
        provenance = Path(model_path) / 'download_provenance.json'
        if provenance.exists():
            self.generation_config['model_provenance'] = json.loads(provenance.read_text())
        tok_mode = 'auto'
        # tensor parallelism follows the number of GPUs given (e.g. "0,1,2,3" for a 32B model); one GPU -> unchanged
        tp = max(VLLM_TENSOR_PARALLEL, len([g for g in str(gpu_ids).split(',') if g.strip()]))
        self.generation_config['tensor_parallel_size'] = tp
        # VLLM_MAX_NUM_SEQS: cap concurrent sequences (vLLM default 256 OOMs the sampler warm-up for a 32B model on 4x24 GB);
        # our batches hold at most BATCH_SIZE=32 prompts, so 32 does not limit throughput. Unset -> vLLM default.
        extra = {}
        if os.environ.get('VLLM_MAX_NUM_SEQS'):
            extra['max_num_seqs'] = int(os.environ['VLLM_MAX_NUM_SEQS'])
            self.generation_config['max_num_seqs'] = extra['max_num_seqs']
        self.llm = LLM(
            model=model_path, tensor_parallel_size=tp,
            trust_remote_code=True, max_model_len=self.generation_config['max_model_len'],
            seed=self.generation_config['seed'], gpu_memory_utilization=0.90, tokenizer_mode=tok_mode, **extra,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.generation_config['tokenizer_mode'] = tok_mode
        self.generation_config['chat_template_sha256'] = hashlib.sha256(
            str(getattr(self.tokenizer, 'chat_template', None)).encode()).hexdigest()
        self.last_batch_metadata = []

    def batch_chat(self, batch_messages, max_tokens=None, stop=None):
        from vllm import SamplingParams
        cfg = self.generation_config
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        # A stop word inside <think> must not cut off the actual action.
        if stop is None:
            stop = [] if cfg['enable_thinking'] else ['Observation:', 'observation:']
        template_args = {}
        if cfg['enable_thinking'] is not None:
            template_args['enable_thinking'] = cfg['enable_thinking']
        prompts = [self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **template_args,
        ) for msgs in batch_messages]
        prompt_lengths = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        if any(n + max_tokens > cfg['max_model_len'] for n in prompt_lengths):
            raise ValueError('Full history plus generation budget exceeds context; do not silently truncate')
        seeds = [((stable_seed(SEED, 'replicate', REPLICATE, p) if REPLICATE else stable_seed(SEED, p)) % (2**31))
                 if cfg['seed_policy'] == 'prompt_sha256' else cfg['seed'] for p in prompts]
        params = [SamplingParams(
            temperature=cfg['temperature'], top_p=cfg['top_p'], top_k=cfg['top_k'],
            max_tokens=max_tokens, stop=stop, seed=seed,
        ) for seed in seeds]
        outputs = self.llm.generate(prompts, params)
        self.last_batch_metadata = [{
            'prompt_tokens': n, 'completion_tokens': len(out.outputs[0].token_ids),
            'finish_reason': out.outputs[0].finish_reason,
            'stop_reason': out.outputs[0].stop_reason,
            'truncated': out.outputs[0].finish_reason == 'length', 'seed': seed,
            'max_tokens': max_tokens,
        } for out, n, seed in zip(outputs, prompt_lengths, seeds)]
        return [(out.outputs[0].text.strip(), len(out.outputs[0].token_ids)) for out in outputs]


class BatchAnthropicLLM:
    """Official Anthropic Messages API backend with the batch_chat interface.

    Sonnet 5 / Opus 5 reject sampling parameters, so decoding is the server default (not temperature 0);
    this is recorded in generation_config. The shared system prompt is marked cacheable.
    """

    def __init__(self, model_name: str, thinking=False, max_tokens=None, concurrency=None):
        from anthropic import Anthropic
        concurrency = concurrency or int(os.environ.get('API_CONCURRENCY', '6'))
        key = os.environ.get('ANTHROPIC_API_KEY')
        if not key and os.environ.get('ANTHROPIC_API_KEY_FILE'):
            key = Path(os.path.expanduser(os.environ['ANTHROPIC_API_KEY_FILE'])).read_text().strip()
        if not key:
            raise RuntimeError('ANTHROPIC_API_KEY or ANTHROPIC_API_KEY_FILE must be set')
        self.model = model_name
        self.client = Anthropic(api_key=key, timeout=120.0, max_retries=3)
        self.concurrency = concurrency
        haiku = 'haiku' in model_name
        self.generation_config = {
            'backend': 'anthropic', 'model_path': 'anthropic:' + model_name, 'enable_thinking': None,
            'temperature': (TEMPERATURE if haiku else 'server_default'), 'top_p': None, 'top_k': None,
            'max_tokens': max_tokens if max_tokens is not None else 1024, 'max_model_len': None,
            'seed': None, 'seed_policy': 'none',
            # conversation caching (ANTHROPIC_CACHE_CONVERSATION, default on): the last message also carries a cache
            # breakpoint, so the next step reads the shared prefix from cache. Outputs are unaffected.
            'prompt_caching': 'system_block' if os.environ.get('ANTHROPIC_CACHE_CONVERSATION', '1') == '0'
                              else 'system_block+last_message',
        }
        # Extended thinking (ANTHROPIC_THINKING): 'adaptive' (Claude 4.6+ models such as Sonnet 5) or an integer
        # budget_tokens (earlier models such as Haiku 4.5). Applied to agent-step calls only; short side-channel calls
        # (max_tokens <= 64) never think. Temperature is not set when thinking is on.
        th = os.environ.get('ANTHROPIC_THINKING', '').strip()
        self.thinking = None if not th else ('adaptive' if th == 'adaptive' else int(th))
        if self.thinking is not None:
            self.generation_config.update(enable_thinking=self.thinking, temperature='server_default_thinking')
        self.max_tokens = self.generation_config['max_tokens']
        self.last_batch_metadata = []
        import threading
        self._usage_lock = threading.Lock()
        self.usage_totals = {'calls': 0, 'input_tokens': 0, 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0,
                             'output_tokens': 0, 'errors': 0, 'refusals': 0}

    def _one(self, msgs, max_tokens, stop):
        system = "\n\n".join(m['content'] for m in msgs if m['role'] == 'system')
        chat = [{'role': m['role'], 'content': m['content']} for m in msgs if m['role'] != 'system']
        if chat and self.generation_config['prompt_caching'] == 'system_block+last_message':
            chat[-1] = {'role': chat[-1]['role'],
                        'content': [{'type': 'text', 'text': chat[-1]['content'], 'cache_control': {'type': 'ephemeral'}}]}
        req = dict(model=self.model, max_tokens=max_tokens, messages=chat)
        if system:
            req['system'] = [{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}]
        if stop:
            req['stop_sequences'] = stop[:4]
        if self.thinking is not None and max_tokens > 64:
            if self.thinking == 'adaptive':
                req['thinking'] = {'type': 'adaptive'}
                req['max_tokens'] = max(max_tokens, 8192)
            else:
                req['thinking'] = {'type': 'enabled', 'budget_tokens': self.thinking}
                req['max_tokens'] = max(max_tokens, self.thinking + 1024)
        elif 'haiku' in self.model:
            req['temperature'] = TEMPERATURE
        last_err = None
        import time, random as _r
        for attempt in range(8):   # SDK already retries 3x internally (honours retry-after); this is the outer backoff
            try:
                r = self.client.messages.create(**req)
                text = ''.join(b.text for b in r.content if getattr(b, 'type', '') == 'text')
                u = r.usage
                with self._usage_lock:  # every call, including side-channel judgment calls, for exact cost accounting
                    self.usage_totals['calls'] += 1
                    for k in ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens', 'output_tokens'):
                        self.usage_totals[k] += getattr(u, k, 0) or 0
                    self.usage_totals['refusals'] += r.stop_reason == 'refusal'
                    cost = self.cost_usd()
                cap = os.environ.get('COST_CAP_USD')
                if cap and cost > float(cap):
                    raise CostCapExceeded(f'cost ${cost:.2f} exceeded cap ${cap}')
                meta = {'prompt_tokens': (u.input_tokens or 0) + (getattr(u, 'cache_read_input_tokens', 0) or 0) + (getattr(u, 'cache_creation_input_tokens', 0) or 0),
                        'completion_tokens': u.output_tokens, 'finish_reason': r.stop_reason, 'stop_reason': r.stop_sequence,
                        'truncated': r.stop_reason == 'max_tokens', 'seed': None, 'max_tokens': max_tokens,
                        'returned_model': r.model, 'response_id': r.id,
                        'cache_read_input_tokens': getattr(u, 'cache_read_input_tokens', None)}
                return text.strip(), meta
            except CostCapExceeded:
                raise
            except Exception as e:
                last_err = e
                time.sleep(min(60, 2 ** attempt) + _r.random())
        with self._usage_lock:
            self.usage_totals['errors'] += 1
        return '', {'prompt_tokens': None, 'completion_tokens': None, 'finish_reason': 'error', 'stop_reason': None,
                    'truncated': False, 'seed': None, 'max_tokens': max_tokens, 'error': repr(last_err)[:300]}

    def cost_usd(self):
        # $/MTok (provider list prices): Haiku 4.5 1/5; Sonnet 5 and 5.5 2/10; others 3/15
        pin, pout = (1.0, 5.0) if 'haiku' in self.model else ((2.0, 10.0) if self.model in ('claude-sonnet-5', 'claude-sonnet-5-5') else (3.0, 15.0))
        u = self.usage_totals
        return ((u['input_tokens'] + 1.25 * u['cache_creation_input_tokens'] + 0.1 * u['cache_read_input_tokens']) * pin
                + u['output_tokens'] * pout) / 1e6

    def batch_chat(self, batch_messages, max_tokens=None, stop=None):
        from concurrent.futures import ThreadPoolExecutor
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if stop is None:
            stop = ['Observation:', 'observation:']
            if os.environ.get('AGENT_PARSE_XML_TOOLCALLS') == '1':
                stop = stop + ['</function_calls>']   # amendment 20: end the turn after the first native tool call
        with ThreadPoolExecutor(self.concurrency) as ex:
            outs = list(ex.map(lambda m: self._one(m, max_tokens, stop), batch_messages))
        errs = [m for _, m in outs if m.get('finish_reason') == 'error']
        if errs and os.environ.get('API_FAIL_LOUD', '1') == '1':
            # never let an API failure silently become a 'format failure' or an 'unparsed' judgment
            raise RuntimeError(f'{len(errs)} API call(s) failed after retries: {errs[0].get("error")}')
        if os.environ.get('AGENT_PARSE_XML_TOOLCALLS') == '1':
            # amendment 20: native tool-call markup is not caught by stop sequences, so the model may write several
            # invented calls in one turn; keep the turn up to the end of its first call and execute only that call.
            cut = []
            for text, m in outs:
                i = text.find('</invoke>')
                if i >= 0:
                    tail = text[i + len('</invoke>'):]
                    if tail.strip():
                        m = dict(m, truncated=False, cut_after_first_toolcall=True, cut_chars=len(tail))
                        text = text[:i + len('</invoke>')] + '\n</function_calls>'
                cut.append((text, m))
            outs = cut
        self.last_batch_metadata = [m for _, m in outs]
        return [(text, (m['completion_tokens'] or 0)) for text, m in outs]


class CostCapExceeded(RuntimeError):
    pass


def make_llm(model_path_or_key: str, gpu_ids: str = '0', thinking=False, max_tokens=None):
    """Factory: 'anthropic:<name>' -> BatchAnthropicLLM (official API), else a local vLLM model."""
    if str(model_path_or_key).startswith('anthropic:'):
        return BatchAnthropicLLM(model_path_or_key[10:], thinking, max_tokens)
    return BatchLocalLLM(model_path_or_key, gpu_ids, thinking, max_tokens)
