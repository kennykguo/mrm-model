// Keep every node and edge; spacing controls geometry, never record counts.
const WIDTH=320,COLUMN=420,PADDING=80,GROUP_COLUMNS=8;

export function graphLabel(n) {
  return n.family==='function' ? n.qualified.replace(n.module+'.','') : n.name;
}

function height(n) {
  const value=['leaf','setting','derived'].includes(n.type)&&n.value?22+Math.ceil(n.value.length/38)*16:0;
  return 112+Math.ceil(graphLabel(n).length/34)*20+(n.symbol?40:0)+value;
}

export function graphEdges(graph) {
  const unique=new Map();
  const add=e=>{
    if(!graph.nodes.has(e.from)||!graph.nodes.has(e.to))return;
    unique.set(JSON.stringify([e.from,e.to,e.kind]),e);
  };
  for(const edges of graph.incoming.values())for(const edge of edges)add(edge);
  for(const n of graph.nodes.values())for(const source of n.source_ids||[])add({from:source,to:n.id,kind:'source definition'});
  return [...unique.values()];
}

function category(n) {
  if(n.family==='model assumption'||n.family==='numerical method')return '00 · model assumptions & numerical methods';
  if(['equation','physical constant','mathematical constant'].includes(n.family))return '01 · mathematical definitions';
  if(n.physics)return '02 · physics · '+(n.module||n.qualified?.split('.').slice(0,-1).join('.')||n.group||n.type);
  if(n.family==='function')return '03 · functions · '+n.module;
  if(n.family==='code constant')return '04 · code constants · '+n.module;
  return '05 · model & simulation · '+(n.group||n.type);
}

export function layoutUniverse(graph) {
  const categories=new Map();
  for(const n of graph.nodes.values()) {
    const key=category(n);if(!categories.has(key))categories.set(key,[]);categories.get(key).push(n);
  }
  const groupWidth=GROUP_COLUMNS*COLUMN-80;
  const totalHeight=[...categories.values()].reduce((sum,nodes)=>sum+Math.ceil(nodes.length/GROUP_COLUMNS)*(Math.max(...nodes.map(height))+80)+200,0);
  const lanes=Math.max(1,Math.ceil(Math.sqrt(totalHeight*1.6/(groupWidth+180))));
  const groups=[],positions=new Map(),bottoms=Array(lanes).fill(240);
  for(const [name,nodes]of [...categories].sort((a,b)=>a[0].localeCompare(b[0]))) {
    nodes.sort((a,b)=>graphLabel(a).localeCompare(graphLabel(b))||a.id.localeCompare(b.id));
    const lane=bottoms.indexOf(Math.min(...bottoms)),rowHeight=Math.max(...nodes.map(height))+80;
    const group={id:name,name,x:PADDING+lane*(groupWidth+180),y:bottoms[lane],width:groupWidth,
      height:Math.ceil(nodes.length/GROUP_COLUMNS)*rowHeight+100,rowHeight,count:nodes.length};
    groups.push(group);
    nodes.forEach((n,index)=>positions.set(n.id,{id:n.id,x:group.x+(index%GROUP_COLUMNS)*COLUMN,
      y:group.y+80+Math.floor(index/GROUP_COLUMNS)*rowHeight,width:WIDTH,height:height(n),row:Math.floor(index/GROUP_COLUMNS),group}));
    bottoms[lane]+=group.height+100;
  }
  return {positions,groups,width:PADDING*2+lanes*(groupWidth+180),height:Math.max(...bottoms)+80};
}

export function layoutDerivation(graph,ids,edges) {
  const nodes=ids.map(id=>graph.nodes.get(id)),rank=new Map(),parents=new Map(ids.map(id=>[id,[]])),visiting=new Set();
  for(const e of edges)if(parents.has(e.to)&&parents.has(e.from))parents.get(e.to).push(e.from);
  function depth(id) {
    if(rank.has(id))return rank.get(id);
    if(visiting.has(id))return 0;
    visiting.add(id);
    const inputs=parents.get(id),level=inputs.length?1+Math.max(...inputs.map(depth)):0;
    visiting.delete(id);rank.set(id,level);return level;
  }
  for(const id of ids)depth(id);
  const columns=new Map(),positions=new Map(),rowHeight=Math.max(...nodes.map(height))+100;
  const group={id:'derivation',name:'complete selected derivation · inputs → result',x:PADDING,y:80,rowHeight,count:nodes.length};
  for(const id of ids){const level=rank.get(id);if(!columns.has(level))columns.set(level,[]);columns.get(level).push(id);}
  for(const [level,list]of [...columns].sort((a,b)=>a[0]-b[0])) {
    list.sort((a,b)=>{
      const centre=id=>{const p=parents.get(id).map(x=>positions.get(x)?.row).filter(x=>x!==undefined);return p.length?p.reduce((a,b)=>a+b,0)/p.length:0;};
      return centre(a)-centre(b)||graphLabel(graph.nodes.get(a)).localeCompare(graphLabel(graph.nodes.get(b)));
    });
    list.forEach((id,row)=>positions.set(id,{id,x:PADDING+level*COLUMN,y:160+row*rowHeight,width:WIDTH,height:height(graph.nodes.get(id)),row,group}));
  }
  group.width=(Math.max(...rank.values())+1)*COLUMN;
  group.height=Math.max(...[...columns.values()].map(x=>x.length))*rowHeight+100;
  return {positions,groups:[group],width:group.width+160,height:group.height+160};
}

export function routeEdge(edge,layout,ordinal=0) {
  const a=layout.positions.get(edge.from),b=layout.positions.get(edge.to);if(!a||!b)return null;
  const start={x:a.x+a.width,y:a.y+a.height/2},end={x:b.x,y:b.y+b.height/2};
  let points;
  if(a.group===b.group) {
    const corridor=a.group.y+80+(a.row+1)*a.group.rowHeight-30;
    points=[start,{x:start.x+18,y:start.y},{x:start.x+18,y:corridor},{x:end.x-18,y:corridor},{x:end.x-18,y:end.y},end];
  } else {
    const ga=a.group,gb=b.group,sa=ga.x+ga.width+45,tb=gb.x+gb.width+75,top=35+(ordinal%70)*2;
    points=[start,{x:start.x+18,y:start.y},{x:start.x+18,y:ga.y+55},{x:sa,y:ga.y+55},{x:sa,y:top},
      {x:tb,y:top},{x:tb,y:gb.y+55},{x:end.x-18,y:gb.y+55},{x:end.x-18,y:end.y},end];
  }
  return {points,path:points.map((p,i)=>`${i?'L':'M'} ${p.x} ${p.y}`).join(' ')};
}

export function boundsOf(layout,ids) {
  const positions=ids.map(id=>layout.positions.get(id)).filter(Boolean);if(!positions.length)return null;
  const x=Math.min(...positions.map(n=>n.x)),y=Math.min(...positions.map(n=>n.y));
  return {x,y,width:Math.max(...positions.map(n=>n.x+n.width))-x,height:Math.max(...positions.map(n=>n.y+n.height))-y};
}
