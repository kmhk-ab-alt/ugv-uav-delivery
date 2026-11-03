"""
共通インターフェース（知覚・判断・制御の“契約”）。
ここを境界にしておくことで、実装を差し替えてROS2/Gazeboフル構成へ拡張できる。

  Perception  … 観測を得る        今:MuJoCoカメラ   将来:ROS2 /camera/image_raw を購読
  Planner     … 次の行動を決める    今:ローカルVLM    将来:学習方策(LeRobot) など
  Controller  … ロボットを動かす    今:MuJoCo PD制御  将来:ros2_control へ指令を publish

このファイルはフレームワーク非依存（MuJoCoもROS2も import しない）。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Observation:
    """1時刻の観測。将来ROS2ではトピックから埋める。"""
    image: Optional[np.ndarray] = None       # HxWx3 RGB（知覚入力）
    joint_pos: Optional[np.ndarray] = None   # 関節角（状態観測 o_state）
    joint_vel: Optional[np.ndarray] = None


@dataclass
class Command:
    """判断層が出す行動指令。今回は離散トリック名＋VLMの判断理由。"""
    trick: str
    reason: str = ""


class Perception(ABC):
    @abstractmethod
    def observe(self) -> Observation: ...


class Planner(ABC):
    @abstractmethod
    def decide(self, obs: Observation) -> Command: ...


class Controller(ABC):
    @abstractmethod
    def hold_stand(self, secs: float) -> bool: ...
    @abstractmethod
    def perform(self, trick: str, dur: float) -> bool: ...
    @abstractmethod
    def running(self) -> bool: ...
