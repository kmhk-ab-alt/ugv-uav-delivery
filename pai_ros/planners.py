"""
判断層。VLMが物体の見た目から「壊す(smash)/避ける(dodge)」を自分で判断する。
色→行動の対応表は持たない。VLMの判断理由はデバッグログ(vlm_debug.log＋コンソール)に出す。

  VLMReactionPlanner … 画像を見てVLMが行動を決定（本命）
  StubPlanner        … VLMなしのオフライン確認用
  LearnedPolicyPlanner … 将来: 学習方策の置き場
"""
from __future__ import annotations
import io, os, json, base64, logging, datetime
from .interfaces import Planner, Observation, Command

ACTIONS = ("smash", "dodge")

# --- デバッグログ設定（ファイル＋コンソール両方に出す） ---
_LOG = logging.getLogger("vlm")
if not _LOG.handlers:
    _LOG.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    _ch = logging.StreamHandler(); _ch.setFormatter(_fmt); _LOG.addHandler(_ch)

# VLMへの指示: 見た目から種別を判断させる（弱いモデルでも従いやすい一語＋理由形式）
_PROMPT = (
    "You control a quadruped robot walking forward. Look at what is in front of it and "
    "decide the action based ONLY on what you see.\n"
    "Rule: if it is an INANIMATE obstacle blocking the path (a wall, bricks, boxes) "
    "-> smash it to break through.\n"
    "If it is a PERSON or a living creature -> dodge it (never hit people).\n"
    "Answer in exactly this format:\n"
    "action: <smash or dodge>\n"
    "reason: <short reason about what you see>"
)


class StubPlanner(Planner):
    """オフライン用。main が与える forced_action をそのまま使う。"""
    def __init__(self):
        self.forced_action = "smash"

    def decide(self, obs: Observation) -> Command:
        return Command(trick=self.forced_action, reason="(stub: VLM未使用)")


class VLMReactionPlanner(Planner):
    """VLMが画像から行動を判断する。複数モデルを順に試すフォールバック付き。
    OpenRouter等へは base_url/api_key を変えるだけ（OpenAI互換）。"""
    def __init__(self, models, base_url: str = "http://localhost:11434/v1",
                 api_key: str = "ollama"):
        # models は単一文字列でもリストでも可
        self.models = [models] if isinstance(models, str) else list(models)
        self.base_url, self.api_key = base_url, api_key

    def _encode(self, image):
        from PIL import Image
        buf = io.BytesIO(); Image.fromarray(image).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _call(self, model, b64):
        """1モデルを呼ぶ。生テキストを返す。失敗時は例外を投げる。"""
        from openai import OpenAI
        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=30.0, max_retries=0)
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}])
        return (resp.choices[0].message.content or "").strip()

    def decide(self, obs: Observation) -> Command:
        if obs.image is None:
            return Command(trick="dodge", reason="(no image)")
        b64 = self._encode(obs.image)
        last = ("dodge", "no model succeeded")
        for model in self.models:
            try:
                raw = self._call(model, b64)
            except Exception as e:
                _LOG.info(f"[VLM] {model} 呼び出し失敗 {type(e).__name__}: {str(e)[:100]} -> 次の候補へ")
                continue
            action, reason = self._parse(raw)
            _LOG.info(f"[VLM] model={model} raw={raw!r}")
            _LOG.info(f"[VLM] model={model} -> action={action} reason={reason}")
            if action in ACTIONS and not reason.startswith("unparsed"):
                return Command(trick=action, reason=f"[{model}] {reason}")
            last = (action, f"[{model}] {reason}")   # 明確でない→次を試す
        _LOG.info(f"[VLM] すべての候補で明確な判断が得られず -> {last[0]}")
        return Command(trick=last[0], reason=last[1])

    @staticmethod
    def _parse(raw: str):
        """最初に現れる smash/dodge を action に、'reason:'以降（無ければ全文）を理由に。"""
        low = raw.lower()
        i_s, i_d = low.find("smash"), low.find("dodge")
        action = None
        if i_s >= 0 and (i_d < 0 or i_s < i_d):
            action = "smash"
        elif i_d >= 0:
            action = "dodge"
        # 理由: 'reason:' 以降を優先、無ければ全文の先頭
        if "reason:" in low:
            reason = raw[low.index("reason:") + len("reason:"):].strip()
        else:
            reason = raw.strip()
        reason = reason[:160]
        if action not in ACTIONS:
            action = "dodge"               # 不明なら安全側（避ける）
            reason = "unparsed:" + reason
        return action, reason


class LearnedPolicyPlanner(Planner):
    """将来の拡張ポイント: LeRobotで学習した方策 π(a|o) をここに載せる。今は未実装。"""
    def decide(self, obs: Observation) -> Command:
        raise NotImplementedError("LeRobot方策を接続する拡張ポイント")
