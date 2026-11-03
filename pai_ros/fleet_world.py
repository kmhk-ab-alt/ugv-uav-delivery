"""MuJoCo visualization with mocap agents and kinematically attached parcels.
Not a flight-dynamics, wheel-drive, or contact-grasp controller.
"""
import math
import numpy as np
import mujoco
from .fleet import COLORS, LANE_SPACING
from .safety import Obstacle
from .neighborhood import ASSETS, environment, parcel_geoms, box, cylinder

class FleetWorld:
    def __init__(self, fleet, headless=True, render=False):
        self.fleet = fleet
        bodies = []
        pads = []
        for a in fleet.agents.values():
            pos = ' '.join(map(str, a.pos))
            if a.kind == 'ground':
                geoms = '<geom type="box" size=".26 .22 .10" rgba=".18 .3 .65 1"/>'
                for x in (-.18, .18):
                    for y in (-.22, .22):
                        geoms += f'<geom type="sphere" pos="{x} {y} -.12" size=".09" rgba=".08 .08 .08 1"/>'
                geoms += '<geom type="box" pos="0 0 .18" size=".18 .18 .025" rgba=".55 .55 .6 1"/>'
                geoms += box((.27,0,.02),(.025,.16,.04),'.87 .91 .89 1')
                geoms += box((.298,0,.03),(.006,.06,.018),'.18 .73 .78 1')
                geoms += box((.05,0,.08),(.13,.15,.04),'.83 .85 .82 1')
                geoms += cylinder((.17,0,.17),.045,.015,'.12 .15 .17 1')
                geoms += f'<camera name="camera_{a.name}" pos="-.6 0 1.05" fovy="25"/>'
            else:
                geoms = '<geom type="box" size=".14 .10 .045" rgba=".15 .17 .2 1"/>'
                for x in (-.21, .21):
                    for y in (-.21, .21):
                        geoms += f'<geom type="capsule" fromto="0 0 0 {x} {y} .02" size=".025" rgba=".3 .35 .39 1" contype="0" conaffinity="0"/>'
                        geoms += f'<geom type="cylinder" pos="{x} {y} .02" size=".12 .01" rgba=".25 .5 .8 .8"/>'
                geoms += '<geom type="box" pos="0 0 -.15" size=".04 .04 .03" rgba=".6 .6 .6 1"/>'
                geoms += f'<camera name="camera_{a.name}" pos="0 0 -.25" fovy="25"/>'
            bodies.append(f'<body name="{a.name}" mocap="true" pos="{pos}">{geoms}</body>')
        for j in fleet.jobs.values():
            pos = ' '.join(map(str, j.pos))
            bodies.append(f'<body name="{j.name}" pos="{pos}"><freejoint name="{j.name}_free"/>' + parcel_geoms(j) + '</body>')
            bodies.append(f'<body name="sling_{j.name}" mocap="true" pos="0 0 -10"><geom name="cable_{j.name}" type="capsule" size=".006 .5" rgba=".8 .8 .8 1" contype="0" conaffinity="0"/></body>')
        xml = ('<mujoco model="PAI neighborhood delivery"><option timestep=".02"/>'
               '<visual><global offwidth="960" offheight="720"/><quality shadowsize="2048"/>'
               '<headlight ambient=".45 .45 .45" diffuse=".6 .6 .6"/>'
               '<map znear=".01" zfar="200"/></visual><asset>' + ASSETS + '</asset>'
               '<worldbody><light pos="1 -4 9" dir="-.3 .2 -1" directional="true" castshadow="true" diffuse=".85 .80 .72"/>'
               + environment(fleet) + ''.join(bodies) + '</worldbody></mujoco>')
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=192, width=192) if render else None
        self.viewer = None
        if not headless:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [.2, (len(fleet.jobs)-1)*LANE_SPACING/2, .4]
            self.viewer.cam.distance = max(16, len(fleet.jobs)*LANE_SPACING*1.15)
            self.viewer.cam.azimuth, self.viewer.cam.elevation = 110, -42
        self.sync()
        if hasattr(fleet.motion,'set_obstacles'):
            fleet.motion.set_obstacles(self.scene_obstacles())

    def scene_obstacles(self):
        obstacles=[]
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid]!=0: continue
            kind=self.model.geom_type[gid]
            if kind==mujoco.mjtGeom.mjGEOM_PLANE: continue
            size=self.model.geom_size[gid].copy()
            if kind==mujoco.mjtGeom.mjGEOM_SPHERE: size[:]=size[0]
            elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER: size[:]=[size[0],size[0],size[1]]
            elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE: size[:]=[size[0],size[0],size[1]+size[0]]
            extent=np.abs(self.data.geom_xmat[gid].reshape(3,3))@size
            center=self.data.geom_xpos[gid]
            if center[2]+extent[2]<=.12: continue # flat road/curb surfaces
            obstacles.append(Obstacle('scene_'+str(gid),tuple(center-extent),tuple(center+extent)))
        return obstacles

    def sync(self):
        for a in self.fleet.agents.values():
            mid = self.model.body(a.name).mocapid[0]
            self.data.mocap_pos[mid] = a.pos
            self.data.mocap_quat[mid] = [math.cos(a.yaw/2),0,0,math.sin(a.yaw/2)]
        for j in self.fleet.jobs.values():
            joint = self.model.joint(j.name + '_free')
            q, v = joint.qposadr[0], joint.dofadr[0]
            self.data.qpos[q:q+3] = j.pos
            self.data.qpos[q+3:q+7] = [1, 0, 0, 0]
            self.data.qvel[v:v+6] = 0
            mid = self.model.body('sling_' + j.name).mocapid[0]
            u = self.fleet.agents.get(j.uav)
            active = u is not None and (j.carrier == u.name or j.phase == 'joint_delivery')
            if active:
                bottom=j.pos[2]+.085
                top=u.pos[2]-.15
                length=max(.015,top-bottom)
                self.data.mocap_pos[mid]=[j.pos[0],j.pos[1],(bottom+top)/2]
                gid=self.model.geom('cable_'+j.name).id
                self.model.geom_size[gid,1]=length/2
                self.model.geom_rbound[gid]=length/2+.006
            else:
                self.data.mocap_pos[mid]=[0,0,-10]
        self.data.time = self.fleet.time
        mujoco.mj_forward(self.model, self.data)
        if self.viewer:
            if hasattr(self.viewer, 'set_texts'):
                rows = [f'{j.name} [P{j.priority}] / {j.package_type} / {j.recipient}: {j.phase}' for j in self.fleet.jobs.values()]
                received = sum(j.receipt_confirmed for j in self.fleet.jobs.values())
                self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    'PAI / PODIUM ODENSE S2 (ADAPTED)\n' + self.fleet.mode.upper() +
                    f'  |  Received {received}/{len(self.fleet.jobs)}  |  {self.fleet.time:.1f}s  |  Warnings {self.fleet.warning_count}  |  Safety holds {getattr(self.fleet.motion,"hold_count",0)}',
                    '\n'.join(rows[:12])))
            self.draw_routes()
            self.viewer.sync()

    def draw_routes(self):
        # Viewer-only overlays are excluded from VLM camera images.
        with self.viewer.lock():
            scene=self.viewer.user_scn
            scene.ngeom=0
            for job in self.fleet.jobs.values():
                for route in job.route_history:
                    color=np.array([.22,.82,1.,.65] if not route['phase'].startswith('return') else [.5,1.,.45,.65],dtype=np.float32)
                    points=route['points']
                    for a,b in zip(points,points[1:]):
                        if scene.ngeom>=scene.maxgeom: return
                        geom=scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),color)
                        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,2.,np.asarray(a,dtype=float),np.asarray(b,dtype=float))
                        scene.ngeom+=1

    def image(self, package):
        if self.renderer is None:
            raise RuntimeError('renderer disabled')
        job = self.fleet.jobs[package]
        observer = job.uav if self.fleet.mode in ('scout', 'joint') else job.ground
        self.renderer.update_scene(self.data, camera='camera_' + observer)
        return np.ascontiguousarray(self.renderer.render()).copy()

    def close(self):
        if self.renderer:
            self.renderer.close()
        if self.viewer:
            self.viewer.close()
