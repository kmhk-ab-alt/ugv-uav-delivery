"""Image-only destination decision. Failure holds the parcel in place."""
import base64
import io
import json
import re
import sys
import time
import numpy as np
from .fleet import COLORS

# Ported from transport_multi.py:classify_vlm, used by transport_fleet.py.
DEFAULT_MODEL = 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free'
DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'
PROMPT = ('An inspection camera looks down at one delivery parcel. '
          'Read the saturated COLOR ROUTING LABEL on TOP of the parcel, not its cardboard, '
          'white barcode label, vehicle, road, or background. '
          'Red routes to the clinic, green to the grocery store, blue to the residential locker. '
          'Answer with exactly one word: red, green, or blue.')


def api_key_for(base_url):
    import os
    from urllib.parse import urlparse
    host = urlparse(base_url).hostname
    key = os.environ.get('OPENROUTER_API_KEY', '')
    if host == 'openrouter.ai':
        if not key:
            raise ValueError('OPENROUTER_API_KEY is required for OpenRouter')
        return key
    return 'ollama' if host in ('localhost', '127.0.0.1', '::1') else key or 'unused'


def parse_decision(raw):
    match = re.search(r'\{[^{}]*\}', raw)
    if match is None:
        raise ValueError('VLM did not return JSON')
    obj = json.loads(match.group())
    if obj.get('destination') not in COLORS:
        raise ValueError('VLM destination is unknown or invalid')
    return obj['destination'], str(obj.get('reason', ''))[:300]

def offline_decide(image):
    # Offline fixture uses actual pixels, never package IDs or scene labels.
    h, w = image.shape[:2]
    arr = image[h//3:2*h//3, w//3:2*w//3].astype(float) / 255
    saturated = arr.max(2) - arr.min(2) > .2
    if not saturated.any():
        raise ValueError('No colored parcel visible')
    rgb = arr[saturated].mean(0)
    return min(COLORS, key=lambda k: np.linalg.norm(rgb - COLORS[k])), 'offline image color classifier'

def safe_detail(text, api_key):
    text = str(text)
    if api_key:
        text = text.replace(api_key, '[API_KEY]')
    text = re.sub(r'sk-[A-Za-z0-9_-]+', '[API_KEY]', text)
    text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '[IMAGE]', text)
    return text.replace('\n', ' ')[:400]

class TemporaryVLMError(RuntimeError):
    """Provider capacity, rate limit, or temporary availability failure."""


def temporary_failure(code, message=''):
    return str(code) in ('408', '429', '500', '502', '503', '504') or any(
        word in message.lower() for word in ('resourceexhausted', 'rate limit', 'worker local total request limit'))

def response_text(resp, preserve_case=False):
    """Validate provider envelopes before indexing completion choices."""
    error = getattr(resp, 'error', None)
    if error is None:
        extra = getattr(resp, 'model_extra', None)
        if isinstance(extra, dict):
            error = extra.get('error')
    if error:
        if isinstance(error, dict):
            code, message = error.get('code', 'unknown'), error.get('message', 'provider error')
            error_type = TemporaryVLMError if temporary_failure(code, str(message)) else ValueError
            raise error_type(f'OpenRouter error code={code}: {message}')
        raise ValueError(f'OpenRouter error: {error}')
    choices = getattr(resp, 'choices', None)
    if not isinstance(choices, (list, tuple)) or not choices:
        raise ValueError('OpenRouter returned no choices (null or empty completion); no color answer')
    choice = choices[0]
    message = getattr(choice, 'message', None)
    content = getattr(message, 'content', None)
    if content is not None and not isinstance(content, str):
        raise ValueError('OpenRouter returned unexpected message content type')
    text=(content or '').strip()
    return text if preserve_case else text.lower(), getattr(choice, 'finish_reason', None)

def _classify_vlm_once(rgb_img, models, base_url, api_key):
    """Port of the existing image -> color classifier, with bounded API calls.

    Preserve the one-word prompt and ordered model fallback. Accept a single
    unambiguous color word; never pick red just because it occurs in a list.
    """
    from PIL import Image
    from openai import OpenAI
    if not models:
        raise ValueError('At least one VLM model is required')
    buf = io.BytesIO()
    Image.fromarray(rgb_img).save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    failures = []
    transient = False
    with OpenAI(base_url=base_url, api_key=api_key, timeout=30., max_retries=0) as client:
        for model in models:
            print(f"[VLM] requesting model={model}", file=sys.stderr, flush=True)
            try:
                resp = client.chat.completions.create(model=model, temperature=0,
                    messages=[{'role': 'user', 'content': [
                        {'type': 'text', 'text': PROMPT},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                    ]}])
                text, finish = response_text(resp)
                words = set(re.findall(r'\b(?:red|green|blue)\b', text))
                if len(words) == 1:
                    color = words.pop()
                    print(f"[VLM] model={model} color={color}", file=sys.stderr, flush=True)
                    return color
                if not text:
                    detail = f'empty content (finish_reason={finish}); no color answer'
                else:
                    detail = 'ambiguous color answer: ' + safe_detail(text, api_key)
                failures.append(f'{model}: {detail}')
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                transient = transient or isinstance(exc, TemporaryVLMError) or temporary_failure(status, str(exc)) or type(exc).__name__ in ('APITimeoutError', 'APIConnectionError')
                detail = type(exc).__name__ + (f' HTTP {status}' if status else '')
                detail += ': ' + safe_detail(str(exc), api_key)
                failures.append(f'{model}: {detail}')
            print('[VLM] ' + failures[-1], file=sys.stderr, flush=True)
    error_type = TemporaryVLMError if transient else RuntimeError
    raise error_type('VLM failed: ' + ' | '.join(failures))

def classify_vlm(rgb_img, models, base_url, api_key):
    # Retry only temporary provider failures; inference runs off the simulation thread.
    for attempt in range(3):
        try:
            return _classify_vlm_once(rgb_img, models, base_url, api_key)
        except TemporaryVLMError:
            if attempt == 2:
                raise
            delay = 2 ** (attempt + 1)
            print(f'[VLM] provider busy; retry {attempt + 1}/2 in {delay}s', file=sys.stderr, flush=True)
            time.sleep(delay)

def vlm_decide(image, models, base_url, api_key):
    color = classify_vlm(image, models, base_url, api_key)
    if color is None:
        raise RuntimeError('No VLM model returned an unambiguous parcel color')
    return color, 'existing transport classifier: ' + color


def priority_decide(image,models=None,base_url='',api_key=None,offline=False):
    """Read importance only. Address is supplied by the order, never by color."""
    from .fleet import PRIORITY_COLORS,PRIORITY_TOKENS
    if offline:
        h,w=image.shape[:2];arr=image[h//3:2*h//3,w//3:2*w//3].astype(float)/255
        mask=arr.max(2)-arr.min(2)>.2
        if not mask.any():raise ValueError('No importance panel visible')
        rgb=arr[mask].mean(0)
        priority=min(PRIORITY_COLORS,key=lambda p:np.linalg.norm(rgb-PRIORITY_COLORS[p]))
        return PRIORITY_TOKENS[priority],'offline importance-color fixture; not address recognition'
    from PIL import Image
    from openai import OpenAI
    buf=io.BytesIO();Image.fromarray(image).save(buf,format='PNG')
    encoded=base64.b64encode(buf.getvalue()).decode();failures=[]
    if not models:raise ValueError('At least one VLM model is required')
    with OpenAI(base_url=base_url,api_key=api_key,timeout=30.,max_retries=0) as client:
        for model in models:
            try:
                print(f'[VLM] importance panel model={model}',file=sys.stderr,flush=True)
                response=client.chat.completions.create(model=model,temperature=0,messages=[{'role':'user','content':[
                    {'type':'text','text':'Read the colored importance panel on the parcel. Red means urgent, yellow means priority, blue means normal. Reply with exactly one word: red, yellow, or blue. Do not infer a delivery address from color.'},
                    {'type':'image_url','image_url':{'url':'data:image/png;base64,'+encoded}}]}])
                answer,_=response_text(response)
                words=set(re.findall(r'\b(?:red|yellow|blue)\b',answer.lower()))
                if len(words)!=1:raise ValueError('No unambiguous importance color')
                return words.pop(),'VLM importance label; address provided by order metadata'
            except Exception as exc:
                failures.append(model+': '+safe_detail(str(exc),api_key))
    raise RuntimeError('Importance recognition failed: '+'; '.join(failures))
