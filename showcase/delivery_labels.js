// Presentation-only diagnostics. Simulator truth never feeds the dispatch model.
const names={red:'赤',green:'緑',blue:'青'};
const colors={red:'#ff5252',green:'#47ed70',blue:'#5599ff'};
export function parcelDisplay(job){
 const id=String(Number(job.name.split('_').pop())+1).padStart(2,'0');
 if(job.numbered_destination){
  const labels={1:'通常・青',2:'優先・黄',3:'緊急・赤'},palette={1:'#5599ff',2:'#ffe04a',3:'#ff5252'};
  return {id,actual:labels[job.priority]||'未確定',recognized:labels[job.recognized_priority]||'確認前',
   target:'配送先'+job.numbered_destination.slice(1),color:palette[job.priority]||'#ffffff',targetColor:'#eeeede',
   status:job.recognized_priority==null?'重要度確認待ち':job.recognized_priority===job.priority?'重要度確認済み':'⚠ 重要度不一致',
   mismatch:job.recognized_priority!=null&&job.recognized_priority!==job.priority};
 }
 const known=Object.hasOwn(names,job.destination);
 return {id,actual:names[job.color]||'不明',recognized:known?names[job.destination]:'確認前',
  target:known?names[job.destination]+'配送先':'配送先未確定',
  color:colors[job.color]||'#ffffff',targetColor:colors[job.destination]||'#ffffff',
  status:known?(job.destination===job.color?'色一致':'⚠ 色不一致'):'認識待ち',
  mismatch:known&&job.destination!==job.color};
}
