"""
MuJoCo実装（シミュレータ・制御・知覚）。
フル構成へ拡張するときは、このファイルの3クラスを
Gazebo/ROS2版に置き換えるだけで main.py は無修正で動く想定。

  MujocoWorld            … 物理世界＋ビューア（環境）     → 将来: Gazebo world
  MujocoGo2Controller    … PD制御でGo2を動かす          → 将来: ros2_control に publish
  MujocoCameraPerception … カメラ描画で画像を得る         → 将来: /camera/image_raw を購読
"""
from __future__ import annotations
import os, time
import numpy as np
import mujoco

from .interfaces import Perception, Controller, Observation
from .go2_tricks import TRICKS, STAND

_SCENE_XML = """<mujoco model="pai go2 react">
  <include file="go2.xml"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="grd" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="grd" texture="grd" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="grd"/>

    <!-- 壊す対象: レンガの壁（道を塞ぐバリケード＝壊して進む対象だと一目で分かる） -->
    <body name="barrier" pos="10 0 0.0">
      <freejoint name="barrier_free"/>
      <geom type="box" size="0.045 0.06 0.05" pos="0 -0.13 0.05" rgba="0.72 0.26 0.18 1" mass="0.03"/>
      <geom type="box" size="0.045 0.06 0.05" pos="0 0    0.05" rgba="0.66 0.30 0.22 1" mass="0.03"/>
      <geom type="box" size="0.045 0.06 0.05" pos="0 0.13 0.05" rgba="0.72 0.26 0.18 1" mass="0.03"/>
      <geom type="box" size="0.045 0.06 0.05" pos="0 -0.065 0.155" rgba="0.68 0.28 0.20 1" mass="0.03"/>
      <geom type="box" size="0.045 0.06 0.05" pos="0 0.065  0.155" rgba="0.68 0.28 0.20 1" mass="0.03"/>
    </body>

    <!-- 避ける対象: 人型（頭・胴・腕・脚のプリミティブ） -->
    <body name="person" pos="10 0 0.03">
      <freejoint name="person_free"/>
      <geom type="capsule" fromto="0 -0.05 0 0 -0.05 0.22" size="0.03" rgba="0.15 0.2 0.55 1"/>
      <geom type="capsule" fromto="0 0.05 0 0 0.05 0.22" size="0.03" rgba="0.15 0.2 0.55 1"/>
      <geom type="capsule" fromto="0 0 0.20 0 0 0.42" size="0.07" rgba="0.85 0.5 0.3 1"/>
      <geom type="capsule" fromto="0 -0.07 0.40 0 -0.13 0.24" size="0.025" rgba="0.85 0.5 0.3 1"/>
      <geom type="capsule" fromto="0 0.07 0.40 0 0.13 0.24" size="0.025" rgba="0.85 0.5 0.3 1"/>
      <geom type="sphere" pos="0 0 0.50" size="0.058" rgba="0.95 0.78 0.62 1"/>
    </body>

    <body name="cam_focus" pos="0.55 0 0.22"/>   <!-- カメラの注視点（前方レーン） -->
    <camera name="cam" pos="-0.08 0 0.55" mode="targetbody" target="cam_focus"/>
  </worldbody>
</mujoco>
"""

# 2種類の対象。VLMが見た目から種別を判断する:
#   barrier = 箱の壁（壊す） / person = 人型（避ける）
VARIANTS = ["barrier", "person"]
# ロボットのホーム姿勢（base7 + 脚12）
HOME19 = np.array([0, 0, 0.27, 1, 0, 0, 0,
                   0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
SPAWN_X, HIDE_X = 0.95, 10.0
BARRIER_SPAWN = np.array([SPAWN_X, 0, 0.0, 1, 0, 0, 0])
BARRIER_HIDE  = np.array([HIDE_X, 0, 0.0, 1, 0, 0, 0])
PERSON_SPAWN  = np.array([SPAWN_X, 0, 0.03, 1, 0, 0, 0])
PERSON_HIDE   = np.array([HIDE_X, 0, 0.03, 1, 0, 0, 0])


class MujocoWorld:
    """環境そのもの。model/data/viewer を保持し、1ステップ進める責務を持つ。"""
    def __init__(self, menagerie_dir: str, headless: bool = True):
        go2_dir = os.path.join(menagerie_dir, "unitree_go2")
        # Resolve includes and meshes without writing into the model checkout.
        import xml.etree.ElementTree as ET
        scene = ET.fromstring(_SCENE_XML)
        scene.find("include").set("file", os.path.join(go2_dir, "go2.xml"))
        compiler = ET.SubElement(scene, "compiler")
        compiler.set("meshdir", os.path.join(go2_dir, "assets"))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self._bar_qadr = self.model.joint("barrier_free").qposadr[0]
        self._bar_dadr = self.model.joint("barrier_free").dofadr[0]
        self._per_qadr = self.model.joint("person_free").qposadr[0]
        self._per_dadr = self.model.joint("person_free").dofadr[0]
        self._active_qadr = self._bar_qadr      # 現在前方にいる対象
        self._active_dadr = self._bar_dadr
        self.viewer = None
        if not headless:
            from mujoco import viewer as _mj_viewer
            self.viewer = _mj_viewer.launch_passive(self.model, self.data)
        self.reset_scene()

    def _place(self, qadr, pose):
        self.data.qpos[qadr:qadr + 7] = pose

    def _zero_vel(self):
        self.data.qvel[self._bar_dadr:self._bar_dadr + 6] = 0
        self.data.qvel[self._per_dadr:self._per_dadr + 6] = 0

    def reset_scene(self):
        """ロボットをホーム姿勢に、両対象を画面外に戻す。"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:19] = HOME19
        self._place(self._bar_qadr, BARRIER_HIDE)
        self._place(self._per_qadr, PERSON_HIDE)
        mujoco.mj_forward(self.model, self.data)

    def set_obstacle(self, variant: str):
        """対象(barrier/person)を前方の出現位置に、もう一方を画面外に置く。"""
        if variant == "barrier":
            self._place(self._bar_qadr, BARRIER_SPAWN); self._place(self._per_qadr, PERSON_HIDE)
            self._active_qadr, self._active_dadr = self._bar_qadr, self._bar_dadr
        else:
            self._place(self._per_qadr, PERSON_SPAWN); self._place(self._bar_qadr, BARRIER_HIDE)
            self._active_qadr, self._active_dadr = self._per_qadr, self._per_dadr
        self._zero_vel()
        mujoco.mj_forward(self.model, self.data)

    def obstacle_x(self) -> float:
        """前方にいる対象の前後位置（x）。小さいほどロボットに近い。"""
        return float(self.data.qpos[self._active_qadr])

    def drive_obstacle(self, vx: float):
        """対象に前後方向の速度を与える（接近させる）。mj_stepの直前に呼ぶ。"""
        self.data.qvel[self._active_dadr] = vx

    def launch_obstacle(self, vx: float, vz: float = 0.0):
        """対象を前方＋上方へ弾き飛ばす（スマッシュ命中の演出）。"""
        self.data.qvel[self._active_dadr] = vx
        self.data.qvel[self._active_dadr + 2] = vz

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            if not self.viewer.is_running():
                return False
            self.viewer.sync()
            time.sleep(self.model.opt.timestep)
        return True

    def running(self):
        return self.viewer is None or self.viewer.is_running()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


class MujocoGo2Controller(Controller):
    """PD制御でGo2を立たせ、トリックを実行する（制御層）。"""
    KP = np.array([60., 60., 80.] * 4)
    KD = np.array([3., 3., 4.] * 4)

    def __init__(self, world: MujocoWorld):
        self.w = world
        self._lo = world.model.actuator_ctrlrange[:, 0]
        self._hi = world.model.actuator_ctrlrange[:, 1]

    def _pd(self, q_des):
        d = self.w.data
        tau = self.KP * (q_des - d.qpos[7:19]) - self.KD * d.qvel[6:18]
        d.ctrl[:] = np.clip(tau, self._lo, self._hi)

    def hold_stand(self, secs: float) -> bool:
        for _ in range(int(secs / self.w.model.opt.timestep)):
            self._pd(STAND)
            if not self.w.step():
                return False
        return True

    def stand_step(self) -> bool:
        """立ち姿勢を保ちながら1ステップだけ進める（VLM推論を待つ間の“生きた待機”用）。"""
        self._pd(STAND)
        return self.w.step()

    def perform(self, trick: str, dur: float = 4.0) -> bool:
        fn = TRICKS.get(trick, TRICKS["bob"])
        for i in range(int(dur / self.w.model.opt.timestep)):
            self._pd(fn(i * self.w.model.opt.timestep, dur))
            if not self.w.step():
                return False
        return True

    def react(self, trick: str, dur: float = 0.9) -> bool:
        """反応動作を実行。smashは命中の瞬間にブロックを弾き飛ばす。"""
        fn = TRICKS.get(trick, TRICKS["bob"])
        n = int(dur / self.w.model.opt.timestep)
        launched = False
        for i in range(n):
            ph = i / n
            if trick == "smash" and not launched and ph >= 0.1:
                self.w.launch_obstacle(vx=4.5, vz=2.5)   # 命中=壁を前方へ吹き飛ばし崩す
                launched = True
            self._pd(fn(i * self.w.model.opt.timestep, dur))
            if not self.w.step():
                return False
        return True

    def running(self) -> bool:
        return self.w.running()


class MujocoCameraPerception(Perception):
    """カメラ描画で観測(画像＋関節状態)を得る（知覚層）。"""
    def __init__(self, world: MujocoWorld, w: int = 320, h: int = 240):
        self.world = world
        self._renderer = mujoco.Renderer(world.model, h, w)

    def observe(self) -> Observation:
        d = self.world.data
        mujoco.mj_forward(self.world.model, d)
        self._renderer.update_scene(d, camera="cam")
        img = self._renderer.render()
        return Observation(image=img,
                           joint_pos=d.qpos[7:19].copy(),
                           joint_vel=d.qvel[6:18].copy())
