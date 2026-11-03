// Presentation is derived from received observations and decisions, never a script of fake outcomes.
export const siteNames={pickup_A:'集荷Aへの道',pickup_B:'集荷Bへの道',meeting_A:'合流Aへの道',meeting_B:'合流Bへの道'};
export function focusJob(s,previous){
 const jobs=Object.values(s.packages||{}),active=j=>j&&!['queued','delivered','failed'].includes(j.phase)&&!j.receipt_confirmed;
 const old=s.packages?.[previous];
 return active(old)?old:jobs.find(j=>active(j)&&j.mode!=='ground')||jobs.find(active)||jobs.find(j=>['return_clear','return_home','dock'].includes(j.phase))||null;
}
export function updateStory(s){
 const $=id=>document.getElementById(id),p=s.site_perception;
 $('vision-label').textContent=p?.mode==='vlm'?'VLM · 実カメラ画像':'テスト画像認識 · VLM未使用';
 $('vision-status').textContent=!p?'地点認識は無効':p.error?'認識エラー・新しい判断を保留':p.pending?'認識中（下の画像・結果は前回分の場合があります）':`観測 #${p.version} → 通行可否 → 地図・経路計画へ`;
 for(const [id,name] of Object.entries(siteNames)){
  const row=p?.sites?.[id],img=$('image-'+id),src=p?.frames?.[id];
  img.hidden=!src;if(src&&img.getAttribute('src')!==src)img.src=src;
  $('label-'+id).textContent=name+' · '+(!p?'未使用':p.pending?'確認中':row?.available?'通行可':'通行不可');
  $('label-'+id).className=p?.pending?'':row?.available?'available':'blocked';
  $('reason-'+id).textContent=p?.pending?'新しい観測を待っています':row?.reason||'画像を待っています';
 }
}
