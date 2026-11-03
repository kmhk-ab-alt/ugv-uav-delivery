import {updateAgentLog} from '/agent_log.js';
import {focusJob,updateStory} from '/story.js';
let storyFocus=null;
import * as T from 'three';
import {parcelDisplay} from '/delivery_labels.js';
import {OrbitControls} from '/vendor/OrbitControls.js';
const $=s=>document.getElementById(s), V=(p)=>new T.Vector3(...p), scene=new T.Scene();
const colors={ground:0x00eaff,uav:0xff4b16}; const parcelTags=new Map();let doc,state=null,cameraMode='auto',lastTime=0,manualUntil=0;
const actors=new Map(),trails=new Map(),labels=[],rotors=[],ropes=new Map();let center=4,centerX=0,extent=12;const overview=()=>{const fit=Math.max(1,1.25/camera.aspect);return {look:new T.Vector3(centerX,center,1),eye:new T.Vector3(centerX+Math.max(13,extent*.65)*fit,center-Math.max(15,extent*.75)*fit,Math.max(12,extent*.8)*fit)};};
let renderer;try{renderer=new T.WebGLRenderer({antialias:true,alpha:false});}catch(e){$('error').style.display='block';$('error').textContent='3D表示に必要なWebGLを開始できませんでした。ブラウザーの設定を確認してください。';throw e;}renderer.setPixelRatio(Math.min(devicePixelRatio,1.6));renderer.setSize(innerWidth,innerHeight);renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.16;document.body.prepend(renderer.domElement);
scene.background=new T.Color('#b4c7cb');scene.fog=new T.Fog('#b4c7cb',28,72);
const camera=new T.PerspectiveCamera(43,innerWidth/innerHeight,.05,150);camera.up.set(0,0,1);camera.position.set(15,-15,14);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.maxPolarAngle=Math.PI*.47;controls.minDistance=1;controls.maxDistance=65;controls.addEventListener('start',()=>{manualUntil=performance.now()+10000;});
scene.add(new T.HemisphereLight(0xdcefff,0x6a785c,2.3));const sun=new T.DirectionalLight(0xffead0,3.2);sun.position.set(9,-12,20);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-22;sun.shadow.camera.right=22;sun.shadow.camera.top=28;sun.shadow.camera.bottom=-20;sun.shadow.normalBias=.025;sun.shadow.bias=-.0002;scene.add(sun);scene.add(sun.target);
const base={box:new T.BoxGeometry(1,1,1),sphere:new T.SphereGeometry(1,14,10),cylinder:new T.CylinderGeometry(1,1,1,18).rotateX(Math.PI/2),plane:new T.PlaneGeometry(1,1)};const mats=new Map();
function material(c){const key=c.join(',');if(!mats.has(key)){const color=new T.Color().setRGB(c[0],c[1],c[2],T.SRGBColorSpace);mats.set(key,new T.MeshStandardMaterial({color,roughness:.77,metalness:.08,transparent:c[3]<.99,opacity:c[3],side:T.DoubleSide,depthWrite:c[3]>.8}));}return mats.get(key);}
function pose(obj,p){obj.position.set(...p.slice(0,3));obj.rotation.set(...p.slice(3,6),'ZYX');}
function sprite(text,color='#c9eef0'){const canvas=document.createElement('canvas');canvas.width=512;canvas.height=96;const c=canvas.getContext('2d');c.fillStyle='#142a35dd';c.beginPath();c.roundRect(2,5,508,82,15);c.fill();c.fillStyle=color;c.font=`500 ${Math.min(31,Math.floor(470/Math.max(1,text.length)))}px sans-serif`;c.textAlign='center';c.fillText(text,256,60);const map=new T.CanvasTexture(canvas);map.colorSpace=T.SRGBColorSpace;const s=new T.Sprite(new T.SpriteMaterial({map,depthTest:false,transparent:true,opacity:.92}));s.scale.set(1.8,.34,1);s.renderOrder=20;return s;}
function addScene(data){center=data.center_y;const ys=data.models.filter(m=>m.name.startsWith('package_')).map(m=>m.pose[1]);extent=Math.max(12,Math.max(...ys)-Math.min(...ys));if(data.bounds){camera.near=.3;scene.fog.near=extent*1.8;centerX=(data.bounds[0]+data.bounds[1])/2;center=(data.bounds[2]+data.bounds[3])/2;extent=Math.max(data.bounds[1]-data.bounds[0],data.bounds[3]-data.bounds[2]);scene.fog.near=extent*1.8;sun.shadow.camera.left=-35;sun.shadow.camera.right=35;sun.shadow.camera.top=35;sun.shadow.camera.bottom=-25;sun.shadow.camera.updateProjectionMatrix();}camera.far=Math.max(150,extent*5);camera.updateProjectionMatrix();controls.maxDistance=Math.max(65,extent*3);scene.fog.far=Math.max(72,extent*4);const staticMeshes=[];
for(const m of data.models){if(m.name==='physical_floor')continue;const group=new T.Group();pose(group,m.pose);group.userData.name=m.name;
 for(const l of m.links){const link=new T.Group();pose(link,l.pose);for(const g of l.objects){let geometry=base[g.kind];if(g.kind==='mesh'){geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.Float32BufferAttribute(g.vertices.flat(),3));geometry.computeVertexNormals();}const mesh=new T.Mesh(geometry,material(g.color));mesh.scale.set(...g.scale);pose(mesh,g.pose);mesh.castShadow=g.kind!=='plane'&&g.color[3]>.8&&!((g.kind==='box'&&g.scale[2]<.08)||(g.kind==='cylinder'&&g.scale[2]<.04));mesh.receiveShadow=true;link.add(mesh);}group.add(link);if(m.name.startsWith('uav_')&&l.name.startsWith('rotor_'))rotors.push({part:link,owner:m.name});}
 const vehicleKind=m.name.startsWith('ground_')?'ground':m.name.startsWith('uav_')?'uav':null;
 if(vehicleKind){
  // Presentation scale only; measured poses, camera input and collision bodies stay unchanged.
  const visual=new T.Group();for(const child of [...group.children])visual.add(child);
  visual.scale.setScalar(1.6);if(vehicleKind==='ground')visual.position.z=.144;
  const accent=vehicleKind==='ground'?0x00efff:0xff6500;
  visual.traverse(o=>{if(o.isMesh){const c=o.material.color;
   if(Math.max(c.r,c.g,c.b)-Math.min(c.r,c.g,c.b)>.15){o.material=o.material.clone();o.material.color.setHex(accent);o.material.emissive.setHex(accent);o.material.emissiveIntensity=.35;o.material.roughness=.4;}
  }});group.add(visual);
  const ring=new T.Mesh(new T.RingGeometry(.68,.76,48),new T.MeshBasicMaterial({color:accent,side:T.DoubleSide,transparent:true,opacity:.85}));
  ring.position.z=vehicleKind==='ground'?-.21:-.1;group.add(ring);
 }
 scene.add(group);const dynamic=/^(ground_|uav_|package_|meeting_|site_blocker_|temporary_)/.test(m.name);if(dynamic)actors.set(m.name,group);else staticMeshes.push(group);
 if(m.name.startsWith('ground_')||m.name.startsWith('uav_')){const kind=m.name.startsWith('uav')?'uav':'ground';const num=Number(m.name.split('_').pop())+1;const tag=sprite((kind==='uav'?'UAV / ':'UGV / ')+String(num).padStart(2,'0'),kind==='uav'?'#ffda99':'#9df0ed');scene.add(tag);labels.push({tag,actor:group,height:kind==='uav'?.75:1.05});}
}
// Batch static scenery by shape and material; shared geometry keeps the browser lightweight.
const batches=new Map();scene.updateMatrixWorld(true);
for(const g of staticMeshes){g.traverse(o=>{if(o.isMesh){const key=o.geometry.uuid+o.material.uuid+o.castShadow;if(!batches.has(key))batches.set(key,{geo:o.geometry,mat:o.material,matrices:[],shadow:o.castShadow});batches.get(key).matrices.push(o.matrixWorld.clone());}});scene.remove(g);}
for(const b of batches.values()){const mesh=new T.InstancedMesh(b.geo,b.mat,b.matrices.length);b.matrices.forEach((m,i)=>mesh.setMatrixAt(i,m));mesh.castShadow=b.shadow;mesh.receiveShadow=true;scene.add(mesh);}
if(data.layout==='blocks'){for(const p of data.places){if(p.name.startsWith('集荷センター')||p.name.startsWith('合流地点'))continue;const tag=sprite(p.name,p.name==='中央公園'?'#aee4b5':'#d9e6df');tag.position.set(...p.position);tag.scale.set(3.3,.62,1);scene.add(tag);}}
for(const site of data.delivery_sites||[]){siteMarker('numbered_'+site.id,'配送先'+site.id.slice(1)+(site.position[2]>.2?' / 山間':' / 市街地'),site.position,'#eeeede');}
const depot=sprite('LOGISTICS HUB / 集荷拠点');depot.position.set(7.8,center,3.4);depot.scale.set(4,.75,1);if(data.layout!=='blocks')scene.add(depot);
const destination=sprite('DELIVERY AREA / 配送先','#ffd899');destination.position.set(-4.5,center,3.4);destination.scale.set(4,.75,1);if(data.layout!=='blocks')scene.add(destination);
if(data.layout==='blocks'){
 const district=sprite('山間の配送先 / 車両進入路なし','#ffd899');district.position.set(-8,22,2);district.scale.set(4.7,.7,1);scene.add(district);
}
controls.target.set(.2,center,1);camera.position.set(13,center-15,12);sun.target.position.set(0,center,0);$('delivered').textContent='0 / '+data.mission_count;
}
function lineBetween(name,points,color,dashed=false){let line=ropes.get(name);if(line){scene.remove(line);line.geometry.dispose();line.material.dispose();}const geo=new T.BufferGeometry().setFromPoints(points.map(V));line=new T.Line(geo,dashed?new T.LineDashedMaterial({color,dashSize:.12,gapSize:.09,transparent:true,opacity:.6}):new T.LineBasicMaterial({color,transparent:true,opacity:.8}));if(dashed)line.computeLineDistances();scene.add(line);ropes.set(name,line);}
// Delivery landmarks use observed mission data, never a hidden parcel label.
const siteMarkers=new Map();
function siteMarker(key,text,position,color,visible=true){
 let item=siteMarkers.get(key);
 if(!item){
  const tag=sprite(text,color);tag.scale.set(2.3,.43,1);
  const ring=new T.Mesh(new T.RingGeometry(key.startsWith('destination_')?.12:.32,key.startsWith('destination_')?.15:.43,32),new T.MeshBasicMaterial({color,side:T.DoubleSide,depthWrite:false}));
  scene.add(tag,ring);item={tag,ring,text};siteMarkers.set(key,item);
 }
 if(item.text!==text){scene.remove(item.tag);item.tag.material.map.dispose();item.tag.material.dispose();item.tag=sprite(text,color);item.tag.scale.set(2.3,.43,1);scene.add(item.tag);item.text=text;}
 item.ring.material.color.set(color);
 item.tag.position.set(position[0],position[1],position[2]+.77);
 item.ring.position.set(position[0],position[1],position[2]-.02);
 item.tag.visible=item.ring.visible=visible;
}
function deliverySites(jobs){
 if(doc.layout!=='blocks')return;
 for(const [i,p] of (doc.pickup_sites||[]).entries()){
  const busy=jobs.some(j=>['preflight','approach','observe','ground_pickup'].includes(j.phase)&&Math.abs(j.source[1]-p[1])<3);
  siteMarker('pickup_center_'+i,'集荷センター '+(i?'B':'A')+(state.site_perception?.pending?' / 認識中':state.site_perception&&!state.site_perception.sites['pickup_'+'AB'[i]]?.available?' / 進入路通行止め':busy?' / 使用中':' / 空き'),p,'#89eadb');
  siteMarkers.get('pickup_center_'+i).tag.material.opacity=busy?1:.7;
 }
 for(const [i,p] of (doc.meeting_sites||[]).entries()){
  const busy=jobs.some(j=>(['to_transfer','handoff','uav_lift','uav_delivery','uav_drop','lower_package','receiving'].includes(j.phase)||(j.phase==='return_clear'&&state.agents[j.ground]&&Math.hypot(state.agents[j.ground].pos[0]-p[0],state.agents[j.ground].pos[1]-p[1])<1.1))&&j.meeting_revision>0&&Math.hypot(j.transfer[0]-p[0],j.transfer[1]-p[1])<1.1);
  siteMarker('meeting_center_'+i,'合流地点 '+(i?'B':'A')+(state.site_perception?.pending?' / 認識中':state.site_perception&&!state.site_perception.sites['meeting_'+'AB'[i]]?.available?' / 進入路通行止め':busy?' / 予約・使用中':' / 空き'),p,'#ffe29e');
  siteMarkers.get('meeting_center_'+i).tag.material.opacity=busy?1:.7;
 }
 for(const j of jobs){
  const n=String(Number(j.name.split('_').pop())+1).padStart(2,'0');
  const active=!['queued','delivered','failed'].includes(j.phase)&&!j.receipt_confirmed;
  // Parcel slots belong to two collection centers; label facilities once.
  // Rendezvous facility labels remain fixed; reservations are shown below.
  const destination=j.destination&&j.delivery_goals?.[j.destination];
  if(destination&&!j.numbered_destination)siteMarker('destination_'+j.name,parcelDisplay(j).target+' '+n+' / '+(j.mode==='ground'?'UGV':'UAV'),destination,parcelDisplay(j).targetColor,active||j.receipt_confirmed);
 }
}
function stateUpdate(s,events,age){state=s;const jobs=Object.values(s.packages),agents=Object.values(s.agents),received=jobs.filter(j=>j.receipt_confirmed).length;
$('delivered').textContent=`${received} / ${jobs.length}`;$('clock').textContent=`${String(Math.floor(s.time/60)).padStart(2,'0')}:${String(Math.floor(s.time%60)).padStart(2,'0')}`;
budgetUpdate(s,jobs);deliverySites(jobs);
for(const [name,pos] of Object.entries(s.site_blockers||{})){const actor=actors.get('site_blocker_'+name);if(actor)actor.userData.target=V(pos);}
storyFocus=focusJob(s,storyFocus?.name);updateStory(s);updateAgentLog(s,events);
if(age>2)$('log-status').textContent='通信待ち — 最後に受信した状態を表示しています';
for(const label of labels){
 const a=s.agents[label.actor.userData.name];if(!a)continue;
 const loaded=jobs.find(j=>j.carrier?.split('+').includes(a.name));
 const assigned=a.task&&s.packages[a.task];
 const id=j=>String(Number(j.name.split('_').pop())+1).padStart(2,'0');
 const vehicle=(a.kind==='ground'?'UGV ':'UAV ')+String(Number(a.name.split('_').pop())+1).padStart(2,'0');
 const text=loaded?`${vehicle} / 荷物${id(loaded)} ${parcelDisplay(loaded).actual} → ${parcelDisplay(loaded).target} ${parcelDisplay(loaded).mismatch?'⚠':''}`:assigned?`${vehicle} / 担当 ${id(assigned)}・空荷`:`${vehicle} / 待機・空荷`;
 if(text!==label.text){scene.remove(label.tag);label.tag.material.map.dispose();label.tag.material.dispose();label.tag=sprite(text,a.kind==='ground'?'#00eeff':'#ff8866');label.tag.scale.set(3.6,.44,1);scene.add(label.tag);label.text=text;}
}
for(const a of agents){const group=actors.get(a.name);if(!group)continue;group.userData.target=V(a.pos);if(a.orientation?.length===4)group.userData.quat=new T.Quaternion(...a.orientation);else group.userData.quat=new T.Quaternion().setFromAxisAngle(new T.Vector3(0,0,1),a.yaw||0);
 const hist=trails.get(a.name)||[];if(!hist.length||hist[hist.length-1].distanceTo(V(a.pos))>.12){hist.push(V(a.pos));if(hist.length>180)hist.shift();trails.set(a.name,hist);lineBetween('trail_'+a.name,hist.map(v=>v.toArray()),colors[a.kind]);}}
if(doc.layout==='blocks'){for(const a of agents.filter(a=>a.kind==='ground')){const path=s.ground_routes?.[a.name];if(a.task&&path?.length>1)lineBetween('streetplan_'+a.name,path.map(p=>[p[0],p[1],.06]),0x7af1df,true);else{const old=ropes.get('streetplan_'+a.name);if(old){scene.remove(old);old.geometry.dispose();old.material.dispose();ropes.delete('streetplan_'+a.name);}}}}
for(const j of jobs){const group=actors.get(j.name);if(group)group.userData.target=V(j.pos);const mark=actors.get('meeting_'+j.name);if(mark){mark.position.set(j.transfer[0],j.transfer[1],.025);mark.visible=j.meeting_revision>0&&!j.receipt_confirmed;}
 if(j.carrier?.startsWith('uav_')){const a=s.agents[j.carrier];lineBetween('cable_'+j.name,[[a.pos[0],a.pos[1],a.pos[2]-.1],j.pos],0xf5e2b8);}else if(ropes.has('cable_'+j.name)){const r=ropes.get('cable_'+j.name);scene.remove(r);r.geometry.dispose();r.material.dispose();ropes.delete('cable_'+j.name);}
 if(j.phase==='to_transfer'&&j.meeting_revision){const g=s.agents[j.ground],u=s.agents[j.uav];lineBetween('plan_g_'+j.name,s.ground_routes?.[j.ground]||[g.pos,[j.transfer[0],j.transfer[1],.3]],0x72e1dc,true);lineBetween('plan_u_'+j.name,[u.pos,[j.transfer[0],j.transfer[1],1.8]],0xffd193,true);}else{for(const prefix of ['plan_g_','plan_u_']){const r=ropes.get(prefix+j.name);if(r){scene.remove(r);r.geometry.dispose();r.material.dispose();ropes.delete(prefix+j.name);}}}}
}
function priority(j){return (j.priority||1)*3 + (j.phase==='handoff'?100:j.phase==='to_transfer'?80:j.phase==='lower_package'||j.phase==='receiving'?70:j.phase==='uav_delivery'?65:30);}
function cameraTarget(now){if(cameraMode==='vehicle'&&state){const a=Object.values(state.agents).find(a=>a.kind==='ground');if(a)return {look:V(a.pos).add(new T.Vector3(0,0,.12)),eye:V(a.pos).add(new T.Vector3(1.8,-2,1.2))};}if(cameraMode==='overview'||!state)return overview();
let jobs=Object.values(state.packages).filter(j=>j.phase!=='delivered');if(cameraMode==='handoff'){const j=jobs.find(j=>j.phase==='handoff'||j.phase==='to_transfer')||jobs[0];if(j)return {look:V(j.transfer).add(new T.Vector3(0,0,.7)),eye:V(j.transfer).add(new T.Vector3(3,-4,2.6))};}
const j=storyFocus||jobs.sort((a,b)=>priority(b)-priority(a))[0];if(!j)return overview();
if(['ground_delivery','handoff','ground_pickup','observe'].includes(j.phase)){const p=state.agents[j.ground].pos;return {look:V(p).add(new T.Vector3(0,0,.3)),eye:V(p).add(new T.Vector3(3.5,-4,2.5))};}
if(j.phase.startsWith('uav_')||j.phase==='lower_package'){const p=state.agents[j.uav].pos;return {look:V(p),eye:V(p).add(new T.Vector3(4,-5,3))};}
if(j.phase==='to_transfer'){const p=state.agents[j.ground].pos;return {look:V(p).lerp(V(j.transfer),.3).add(new T.Vector3(0,0,1)),eye:V(p).add(new T.Vector3(6,-7,5))};}
return overview();}
function frame(now){requestAnimationFrame(frame);if(now-lastTime<33)return;const dt=Math.min(.05,(now-lastTime)/1000||.016);lastTime=now;
for(const g of actors.values()){if(g.userData.target)g.position.lerp(g.userData.target,1-Math.exp(-dt*18));if(g.userData.quat)g.quaternion.slerp(g.userData.quat,1-Math.exp(-dt*18));}
for(const r of rotors){const a=state?.agents?.[r.owner];if(a&&a.pos[2]>.13)r.part.rotation.z+=dt*35;}
for(const l of labels){l.tag.position.copy(l.actor.position).add(new T.Vector3(0,0,l.height));}
if(now>manualUntil){const t=cameraTarget(now);camera.position.lerp(t.eye,1-Math.exp(-dt*.65));controls.target.lerp(t.look,1-Math.exp(-dt*.9));}
controls.update();renderer.render(scene,camera);}
window.addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
for(const b of document.querySelectorAll('[data-camera]'))b.onclick=()=>{cameraMode=b.dataset.camera;manualUntil=0;document.querySelectorAll('[data-camera]').forEach(x=>x.classList.toggle('on',x===b));};
async function poll(){
 try{
  const live=await (await fetch('/api/live',{cache:'no-store',signal:AbortSignal.timeout(4000)})).json();
  if(live.preview)$('log-status').textContent='外観プレビュー — 配送は実行していません';
  else if(live.state)stateUpdate(live.state,live.events,live.age_s);
  else $('log-status').textContent='ROSの配送状態を待っています';
 }catch(e){$('log-status').textContent=state?.complete?'試行終了 — 記録表示':'通信が途切れています — シミュレーターの接続を確認';}
 setTimeout(poll,350);
}
try{doc=await (await fetch('/api/scene')).json();addScene(doc);requestAnimationFrame(frame);setTimeout(poll,1600);}catch(e){$('error').style.display='block';$('error').textContent='3D表示を開始できませんでした。ブラウザーのWebGL設定を確認してください。 '+e.message;}
window.addEventListener('error',e=>{if(!doc){$('error').style.display='block';$('error').textContent=e.message;}});

function budgetUpdate(s,jobs){
 const b=s.service_budget;if(!b){$('scorebox').style.display='none';return;}
 $('clock').textContent=`${String(Math.floor(Math.ceil(b.remaining_s)/60)).padStart(2,'0')}:${String(Math.ceil(b.remaining_s)%60).padStart(2,'0')}`;$('clocklabel').textContent='残り時間';
 $('scorebox').style.display='block';$('score').textContent=`${b.score} pt`;
 for(const j of jobs){const d=parcelDisplay(j),text=`荷物${d.id} ${d.actual} / P${j.priority}`;let tag=parcelTags.get(j.name);if(!tag){tag=sprite(text,d.color);tag.scale.set(1.6,.30,1);scene.add(tag);parcelTags.set(j.name,tag);}const a=actors.get(j.name);if(a)tag.position.copy(a.position).add(new T.Vector3(0,0,.45));tag.visible=!j.receipt_confirmed;}
}
