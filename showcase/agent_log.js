// Logs show observed events and upstream commands, not invented model reasoning.
export const vehicleName=id=>(id?.startsWith('ground_')?'UGV ':'UAV ')+String(Number(id?.split('_').pop())+1).padStart(2,'0');
const cargo=id=>'荷物'+String(Number(id?.split('_').pop())+1).padStart(2,'0');
export function eventRows(e,s){
 const j=s.packages?.[e.package],pair=[j?.ground,j?.uav].filter(Boolean);
 let ids=[],message='';
 switch(e.event){
 case 'command_directive':ids=Object.keys(s.agents||{});message=`司令センター（${e.source==='vlm'?'VLM':'デモルール・VLM未使用'}）：${e.reason}［${(e.priority_order||[]).map(p=>'P'+p).join(' → ')}］`;break;
 case 'dispatch_decision':ids=[e.chosen?.ground,e.chosen?.uav].filter(Boolean);message=`${e.policy==='budget_learned'?'配車MLP':'配車ルール'}：${e.command_directive?`司令センター（${e.command_directive.source==='vlm'?'VLM':'デモルール・VLM未使用'}）の方針に従い、`:''}${cargo(e.package)}（P${e.chosen?.priority}）を選択`;break;
 case 'ground_released':ids=[e.agent];message='受け渡し完了、次の集荷へ';break;
 case 'await_uav':ids=[e.agent];message='先行集荷完了、UAVの帰還を待機';break;
 case 'uav_assigned':ids=[e.agent,e.ground];message='先行集荷した荷物の搬送を開始';break;
 case 'assigned':ids=[e.ground,e.uav].filter(Boolean);message=`配車指示：${cargo(e.package)}を担当`;break;
 case 'decision':ids=[j?.ground].filter(Boolean);message=`画像認識：重要度P${e.recognized_priority}、配送先${e.destination}`;break;
 case 'meeting_selected':ids=pair;message=e.revision>1?(e.site_changed?'道路状況から合流先を変更':'通行止めを避ける経路に更新'):'所要時間を比較して合流先を決定';break;
 case 'flight_plan':ids=[j?.uav].filter(Boolean);message='飛行経路を更新';break;
 case 'obstacle_detour':ids=[e.agent];message='障害物を避ける経路で移動';break;
 case 'collision_hold':ids=[e.agent];message='衝突を避けるため停止';break;
 case 'meeting_arrival':ids=[e.agent];message='合流地点に到着、相手を待機';break;
 case 'handoff_complete':return [{id:e.sender,text:'荷物をUAVへ受け渡し',time:e.time},{id:e.receiver,text:'荷物を受領、配送へ',time:e.time}].filter(r=>r.id);
 case 'receipt_confirmed':ids=[j?.mode==='ground'?j?.ground:j?.uav].filter(Boolean);message='配送先の受領を確認、帰還へ';break;
 case 'delivered':ids=pair;message='帰還完了';break;
 }
 return ids.filter(id=>id&&s.agents?.[id]).map(id=>({id,text:message,time:e.time}));
}
const phaseText={await_uav:'先行集荷済み・UAV待ち',preflight:'出発準備',approach:'集荷へ移動',observe:'荷物を認識中',ground_pickup:'荷物を積載中',to_transfer:'合流地点へ移動',handoff:'荷物を受け渡し中',uav_lift:'荷物を受領、上昇中',uav_delivery:'配送先へ飛行',uav_drop:'配送先へ降下',lower_package:'荷物を降ろす',receiving:'受領確認中',ground_delivery:'配送先へ走行',return_clear:'帰還中',return_home:'帰還中',dock:'着陸・待機',delivered:'配送完了'};
export function currentAction(a,s){
 const j=s.packages?.[a.task];if(!j)return s.ground_returning?.includes(a.name)?'次の指示を待ちながら拠点へ帰還':'待機';
 if(a.kind==='uav'&&['preflight','approach','observe','ground_pickup'].includes(j.phase))return 'UGVの集荷完了を待機';
 if(a.kind==='ground'&&['uav_lift','uav_delivery','uav_drop','lower_package','receiving'].includes(j.phase))return 'UAVの配送完了を待機';
 return phaseText[j.phase]||'指示待ち';
}
const history=new Map(),seen=new Set();let previousTime=-1;
export function updateAgentLog(s,events){
 if(s.time<previousTime){history.clear();seen.clear();}previousTime=s.time;
 for(const e of events){const key=JSON.stringify(e);if(seen.has(key))continue;seen.add(key);
  for(const r of eventRows(e,s)){const list=history.get(r.id)||[];if(list.at(-1)?.text!==r.text)list.push(r);history.set(r.id,list.slice(-6));}
 }
 // Only deduplication of the current server event window is needed.
 if(seen.size>500){seen.clear();for(const e of events)seen.add(JSON.stringify(e));}
 const root=document.getElementById('agent-log');root.replaceChildren(...Object.values(s.agents||{}).map(a=>{
  const section=document.createElement('section');section.className='agent-entry';
  const title=document.createElement('h3');title.textContent=vehicleName(a.name);title.className=a.kind;
  const current=document.createElement('p');current.className='agent-current';current.textContent=currentAction(a,s);
  const list=document.createElement('ol');for(const r of [...(history.get(a.name)||[])].reverse()){
   const li=document.createElement('li'),time=document.createElement('time');time.textContent=`${Math.floor(r.time/60)}:${String(Math.floor(r.time%60)).padStart(2,'0')}`;
   li.append(time,document.createTextNode(r.text));list.append(li);
  }
  section.append(title,current,list);return section;
 }));
 const error=s.errors?.runtime||s.errors?.site_perception;
 const waiting=s.site_perception?.pending;
 document.getElementById('log-status').textContent=s.failed?(error?.includes('lockstep feedback timeout')?'機体位置の受信待ちで停止':error?'認識・制御エラーで停止':'実行を中断'):s.complete?'配送・帰還完了':waiting?(Object.keys(s.site_perception.frames||{}).length?'司令センターが画像を確認中':'カメラ画像を待機中'):'';
}
