"""Versioned episode logs for analysis and future dataset conversion.

These are NOT LeRobot datasets or human demonstrations. Commands are waypoint, body-velocity or compliant-grip setpoints,
not measured motor torques. Physical observations come from Gazebo odometry.
"""
import hashlib
import json
from pathlib import Path
import platform
import uuid

class EpisodeRecorder:
    def __init__(self, root, config):
        self.path=Path(root)/uuid.uuid4().hex
        self.path.mkdir(parents=True)
        self.frames=self.path/'frames';self.frames.mkdir()
        self.writer=(self.path/'steps.jsonl').open('w')
        self.images=(self.path/'observations.jsonl').open('w')
        files=Path(__file__).parent
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files.glob('*.py')}
        manifest=dict(schema_version=1,python=platform.python_version(),platform=platform.platform(),
                      config=config,source_sha256=hashes,
                      action_semantics='body velocity and compliant gripper setpoints; measured odometry' if config.get('physical') else 'scripted position targets, not measured motor commands',
                      clock='simulation seconds; API latency is separate wall time',
                      dataset_format='pai_control_samples_v2 (not LeRobot)' if config.get('physical') else 'pai_episode_v1 (not LeRobot)',physics='gazebo_diffdrive_rotors_compliant_grip' if config.get('physical') else 'kinematic_demo')
        (self.path/'manifest.json').write_text(json.dumps(manifest,indent=2))
        self.physical=bool(config.get('physical'))
        self.index=0

    def step(self, fleet, before):
        # Record every tick so high-rate commands are not mismatched to sparse states.
        if self.physical:
            row=dict(schema_version=2,index=self.index,timestamp=fleet.time,observation=fleet.observation(),action=fleet.motion.commands+fleet.cargo.commands,next_timestamp=None,next_observation=None,semantics='measured state and newly issued setpoints; next-state outcome is not yet observed')
        else:
            row=dict(index=self.index,timestamp=before['time'],next_timestamp=fleet.time,observation=before,action=fleet.motion.commands+fleet.cargo.commands,next_observation=fleet.observation())
        self.writer.write(json.dumps(row)+'\n')
        self.index+=1

    def image(self, fleet, package, pixels):
        from PIL import Image
        filename=f'{self.index:08d}_{package}.png'
        Image.fromarray(pixels).save(self.frames/filename,compress_level=1)
        self.images.write(json.dumps(dict(timestamp=fleet.time,step_index=max(0,self.index-1),
            package=package,image='frames/'+filename,observation=fleet.observation()))+'\n')

    def close(self, fleet, errors=None):
        self.writer.close();self.images.close()
        (self.path/'result.json').write_text(json.dumps(dict(state=fleet.snapshot(),errors=errors or []),indent=2))
