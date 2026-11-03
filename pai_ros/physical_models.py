"""Self-contained Gazebo differential-drive and quadrotor plant models.
Uses Gazebo's DiffDrive, rotor force models and Lee velocity controller.
No external mesh/model download is required.
"""
from .gazebo_scene import visuals
from .neighborhood import box,cylinder
from .presentation_assets import ground_details,drone_details

def inertial(mass,xx,yy,zz):
    return f'<inertial><mass>{mass}</mass><inertia><ixx>{xx}</ixx><iyy>{yy}</iyy><izz>{zz}</izz></inertia></inertial>'
def collision_box(size):
    return f'<collision name="collision"><geometry><box><size>{size}</size></box></geometry></collision>'
def odometry(name,frequency=50):
    return f'<plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher"><dimensions>3</dimensions><odom_publish_frequency>{frequency}</odom_publish_frequency><odom_topic>/model/{name}/odometry</odom_topic></plugin>'
def ground_model(agent):
    name=agent.name
    body=visuals(box((0,0,0),(.24,.19,.09),'.02 .85 .95 1')+box((0,0,.18),(.18,.18,.025),'.55 .55 .6 1')+ground_details())
    out=f'<model name="{name}"><pose>{" ".join(map(str,agent.pos))} 0 0 0</pose><link name="body">{inertial(4,.08,.12,.16)}{collision_box(".48 .38 .18")}{body}</link>'
    for side,y in [('left',.235),('right',-.235)]:
        out+=f'''<link name="{side}_wheel"><pose>0 {y} -.12 1.57079632679 0 0</pose>{inertial(.2,.001,.001,.001)}
<collision name="wheel"><geometry><cylinder><radius>.12</radius><length>.06</length></cylinder></geometry><surface><friction><ode><mu>1</mu><mu2>1</mu2></ode></friction></surface></collision>
{visuals(cylinder((0,0,0),.12,.03,'.08 .08 .08 1'))}</link>
<joint name="{side}_joint" type="revolute"><parent>body</parent><child>{side}_wheel</child><axis><xyz expressed_in="__model__">0 1 0</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit><dynamics><damping>.01</damping></dynamics></axis></joint>'''
    for side,x in [('front',.18),('rear',-.18)]:
        out+=f'''<link name="{side}_caster"><pose>{x} 0 -.15 0 0 0</pose>{inertial(.05,.0001,.0001,.0001)}<collision name="caster"><geometry><sphere><radius>.09</radius></sphere></geometry><surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface></collision>{visuals(f'<geom type="sphere" size=".09" rgba=".15 .15 .15 1"/>')}</link><joint name="{side}_fixed" type="fixed"><parent>body</parent><child>{side}_caster</child></joint>'''
    out+=f'''<plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive"><left_joint>left_joint</left_joint><right_joint>right_joint</right_joint><wheel_separation>.47</wheel_separation><wheel_radius>.12</wheel_radius><topic>/model/{name}/cmd_vel</topic><odom_topic>/model/{name}/wheel_odometry</odom_topic><max_linear_acceleration>1.2</max_linear_acceleration><max_angular_acceleration>3</max_angular_acceleration></plugin>'''
    return out+odometry(name)+'</model>'
def uav_model(agent):
    name=agent.name
    body=box((0,0,0),(.14,.1,.045),'1 .16 .04 1')
    body+=drone_details()
    rotor_positions=[(.21,.21),(-.21,-.21),(.21,-.21),(-.21,.21)]
    for x,y in rotor_positions:body+=f'<geom type="capsule" fromto="0 0 0 {x} {y} .02" size=".02" rgba=".3 .35 .39 1"/>'
    out=f'<model name="{name}"><pose>{" ".join(map(str,agent.pos))} 0 0 0</pose><link name="body">{inertial(1.4,.025,.025,.045)}{collision_box(".28 .20 .09")}{visuals(body)}</link>'
    rotors=''
    for i,(x,y) in enumerate(rotor_positions):
        direction=1 if i<2 else -1
        out+=f'''<link name="rotor_{i}"><pose>{x} {y} .035 0 0 0</pose>{inertial(.025,.00001,.00001,.00002)}{visuals(cylinder((0,0,0),.12,.002,'.12 .17 .20 .24')+box((0,0,.003),(.118,.014,.003),'.08 .12 .15 1')+box((0,0,.004),(.014,.118,.003),'.08 .12 .15 1')+cylinder((0,0,.009),.022,.009,'.28 .33 .36 1'))}</link><joint name="rotor_{i}_joint" type="revolute"><parent>body</parent><child>rotor_{i}</child><axis><xyz>0 0 1</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis></joint>
<plugin filename="gz-sim-multicopter-motor-model-system" name="gz::sim::systems::MulticopterMotorModel"><robotNamespace>{name}</robotNamespace><jointName>rotor_{i}_joint</jointName><linkName>rotor_{i}</linkName><turningDirection>{'ccw' if direction==1 else 'cw'}</turningDirection><timeConstantUp>.0125</timeConstantUp><timeConstantDown>.025</timeConstantDown><maxRotVelocity>1000</maxRotVelocity><motorConstant>8.54858e-06</motorConstant><momentConstant>.016</momentConstant><commandSubTopic>gazebo/command/motor_speed</commandSubTopic><actuator_number>{i}</actuator_number><rotorDragCoefficient>8.06428e-05</rotorDragCoefficient><rollingMomentCoefficient>1e-06</rollingMomentCoefficient><rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim><motorType>velocity</motorType></plugin>'''
        rotors+=f'<rotor><jointName>rotor_{i}_joint</jointName><forceConstant>8.54858e-06</forceConstant><momentConstant>.016</momentConstant><direction>{direction}</direction></rotor>'
    out+=f'''<plugin filename="gz-sim-multicopter-control-system" name="gz::sim::systems::MulticopterVelocityControl"><robotNamespace>{name}</robotNamespace><commandSubTopic>cmd_vel</commandSubTopic><enableSubTopic>enable</enableSubTopic><comLinkName>body</comLinkName><velocityGain>4 4 16</velocityGain><attitudeGain>2 2 .15</attitudeGain><angularRateGain>.3 .3 .15</angularRateGain><maximumLinearAcceleration>4 4 8</maximumLinearAcceleration><rotorConfiguration>{rotors}</rotorConfiguration></plugin>'''
    return out+odometry(name)+'</model>'
def floor_model():
    return '<model name="physical_floor"><static>true</static><link name="ground"><collision name="floor"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></collision></link></model>'
