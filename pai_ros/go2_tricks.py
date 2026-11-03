"""Go2のトリック集（立ったまま・転ばない範囲）。各トリックは
時刻tと継続時間durを受け取り、12関節の目標角(STAND基準のオフセット済み)を返す。
別スクリプトから import して使う。"""
import numpy as np

STAND = np.array([0.0, 0.9, -1.8] * 4)   # 立ち姿勢 (各脚 hip,thigh,calf)

def _bob(t, dur):
    q = STAND.copy()
    a = 0.15 * np.sin(2*np.pi*1.0 * t)
    for leg in range(4):
        q[leg*3+1] += a; q[leg*3+2] += -1.0*a
    return q

def _sway(t, dur):
    q = STAND.copy()
    s = 0.12 * np.sin(2*np.pi*0.6 * t)
    for leg in (0, 2): q[leg*3+1] += s
    for leg in (1, 3): q[leg*3+1] -= s
    return q

def _twist(t, dur):
    q = STAND.copy()
    w = 0.25 * np.sin(2*np.pi*0.6 * t)   # hip外転を前後で逆相→胴体ヨー揺れ
    for leg in (0, 1): q[leg*3+0] += w
    for leg in (2, 3): q[leg*3+0] -= w
    return q

def _stretch(t, dur):
    q = STAND.copy()
    p = 0.5 - 0.5*np.cos(2*np.pi * t/dur)   # ゆっくり沈んで戻る 1回
    a = 0.25 * p
    for leg in range(4):
        q[leg*3+1] += a; q[leg*3+2] += -1.0*a
    return q

def _smash(t, dur):
    """突っ込んで叩き下ろす：頭(前)を大きく下げて前のめりに攻撃（実測で頭下げを確認）。"""
    q = STAND.copy()
    p = np.sin(np.pi * min(t/dur, 1.0))        # 0->1->0
    for leg in (0, 1):                          # 前脚を縮める→前が下がる
        q[leg*3+1] += 0.5*p; q[leg*3+2] += -0.8*p
    for leg in (2, 3):                          # 後脚を伸ばす→後ろが上がる＝前のめり
        q[leg*3+1] += -0.5*p; q[leg*3+2] += 0.8*p
    return q

def _dodge(t, dur):
    """のけぞって後退：頭(前)を大きく上げて後ろに引く（実測で頭上げを確認／対象に触れない）。"""
    q = STAND.copy()
    p = np.sin(np.pi * min(t/dur, 1.0))
    for leg in (0, 1):                          # 前脚を伸ばす→前が上がる
        q[leg*3+1] += 0.2*p; q[leg*3+2] += 0.8*p
    for leg in (2, 3):                          # 後脚を縮める→後ろが沈む＝のけぞり
        q[leg*3+1] += -0.2*p; q[leg*3+2] += -0.8*p
    return q

TRICKS = {"bob": _bob, "sway": _sway, "twist": _twist, "stretch": _stretch,
          "smash": _smash, "dodge": _dodge}

# パネル/物体の色 → どの動作をするか（ロボット側のポリシー）
# 反応ゲーム: 赤いバリア=壊す(smash) / 青いバリア=避ける(dodge)
COLOR_TO_TRICK = {"red": "smash", "blue": "dodge", "green": "bob"}
