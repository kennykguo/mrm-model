import {graphEdges,graphLabel,layoutUniverse,layoutDerivation,boundsOf,routeEdge} from './graph-layout.js';
import {traceGraph,inputsFor} from './derivations.js';

export function createGraphView(graph,{el,math,unit,onSelect,onInspect},initial) {
  const root=el('section',{class:'visual-graph','aria-label':'Interactive map of every indexed definition'});
  const controls=el('div',{class:'map-controls'}),scopes=el('div',{class:'map-scopes','role':'group','aria-label':'Map scope'});
  const all=el('button',{type:'button','aria-pressed':'true'},'all definitions');
  const ancestry=el('button',{type:'button','aria-pressed':'false'},'full selected derivation');scopes.append(all,ancestry);
  const tools=el('div',{class:'map-camera-tools'}),fit=el('button',{type:'button'},'fit map'),focus=el('button',{type:'button'},'focus selected');
  const minus=el('button',{type:'button','aria-label':'Zoom out'},'−'),plus=el('button',{type:'button','aria-label':'Zoom in'},'+');
  tools.append(fit,focus,minus,plus);controls.append(scopes,tools);
  const status=el('p',{class:'map-status note','role':'status'});
  const hint=el('p',{class:'map-hint note'},'drag the empty canvas to pan · scroll or pinch to zoom · click a quantity to inspect it. search highlights matches without removing other nodes.');
  const viewport=el('div',{class:'map-viewport',tabindex:'0','aria-label':'Variable graph canvas'}),world=el('div',{class:'map-world'});
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('class','map-edges');svg.setAttribute('aria-hidden','true');
  svg.innerHTML='<defs><marker id="variable-map-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 1 L 9 5 L 0 9 z" fill="context-stroke"/></marker></defs>';
  world.append(svg);viewport.append(world);
  const mini=el('canvas',{class:'map-minimap',width:'220',height:'140','aria-label':'Overview of all nodes. Click to move the main view.'});viewport.append(mini);
  const preview=el('div',{class:'map-selection-preview','aria-live':'polite'});
  const legend=el('div',{class:'map-legend'});
  for(const [className,text]of [['physical','math & physical quantities'],['assumption','model assumptions & numerical methods'],['implementation','code & simulation'],['connection','arrows point from input to dependent quantity'],['context','dashed arrows show model or source context']])legend.append(el('span',{class:className},text));
  root.append(controls,status,hint,viewport,preview,legend);
  const universe=layoutUniverse(graph),allEdges=graphEdges(graph),buttons=new Map(),paths=[];
  let layout=universe,edges=allEdges,scope='all',selected=initial,camera={x:0,y:0,scale:1},drag=null,matches=new Set();
  let pendingFit=true;
  let previousSize=null;
  const physicsEdge=e=>['equation input','equation constant','defined by','function input','recorded input'].includes(e.kind);

  function cameraChanged() {
    world.style.transform=`translate(${camera.x}px,${camera.y}px) scale(${camera.scale})`;
    viewport.dataset.zoom=String(camera.scale);drawMini();
  }
  function fitBounds(bounds) {
    if(!bounds||!viewport.clientWidth||!viewport.clientHeight)return;
    const margin=60,scale=Math.min((viewport.clientWidth-margin)/(bounds.width+margin),(viewport.clientHeight-margin)/(bounds.height+margin));
    camera={scale,x:(viewport.clientWidth-bounds.width*scale)/2-bounds.x*scale,y:(viewport.clientHeight-bounds.height*scale)/2-bounds.y*scale};cameraChanged();
  }
  function fitMap(){fitBounds({x:0,y:0,width:layout.width,height:layout.height});}
  function focusNode(id) {
    const p=layout.positions.get(id);if(!p)return;
    const scale=Math.min(1.15,(viewport.clientWidth-56)/(p.width+60));
    camera={scale,x:viewport.clientWidth/2-(p.x+p.width/2)*scale,y:viewport.clientHeight/2-(p.y+p.height/2)*scale};cameraChanged();
  }
  function zoom(factor,x=viewport.clientWidth/2,y=viewport.clientHeight/2) {
    const scale=camera.scale*factor;
    camera={scale,x:x-(x-camera.x)*factor,y:y-(y-camera.y)*factor};cameraChanged();
  }
  function drawMini() {
    const context=mini.getContext('2d'),w=mini.width,h=mini.height,pad=6;
    context.clearRect(0,0,w,h);
    const ratio=Math.min((w-pad*2)/layout.width,(h-pad*2)/layout.height);
    context.fillStyle=getComputedStyle(root).getPropertyValue('--bg');context.fillRect(0,0,w,h);
    for(const [id,p]of layout.positions){const n=graph.nodes.get(id);context.fillStyle=id===selected?'#216bba':['model assumption','numerical method'].includes(n.family)?'#b77d30':n.physics?'#167758':'#929292';context.fillRect(pad+p.x*ratio,pad+p.y*ratio,Math.max(1,p.width*ratio),Math.max(1,p.height*ratio));}
    context.strokeStyle='#216bba';context.lineWidth=1.2;context.strokeRect(pad-camera.x/camera.scale*ratio,pad-camera.y/camera.scale*ratio,viewport.clientWidth/camera.scale*ratio,viewport.clientHeight/camera.scale*ratio);
    mini.dataset.ratio=ratio;mini.dataset.padding=pad;
  }

  function selectedPreview() {
    const n=graph.nodes.get(selected);preview.replaceChildren();
    const heading=el('div',{class:'map-preview-heading'});if(n.symbol)heading.append(math(n.symbol));heading.append(el('strong',{},graphLabel(n)),el('span',{class:'kind-badge'},n.family));preview.append(heading);
    if(n.formula_tex)preview.append(math(n.formula_tex,true));
    if(n.value&&n.family!=='code constant')preview.append(el('p',{class:'note'},'indexed reference value — '+n.value));
    if(n.scope)preview.append(el('p',{class:'note'},n.scope));
    else if(n.meaning&&n.meaning!==n.name)preview.append(el('p',{class:'note'},n.meaning));
    if(n.reference_formula){preview.append(el('p',{class:'note'},'indexed reference with backscatter disabled; the TCMT conversion would give'),math(n.reference_formula,true));}
    const footer=el('div',{class:'map-preview-footer'});if(n.unit)footer.append(unit(n.unit));
    const inspect=el('button',{type:'button',class:'text-button'},'definition, assumptions & source ↓');inspect.onclick=onInspect;footer.append(inspect);preview.append(footer);
  }

  function updateHighlights() {
    const related=new Set(traceGraph(graph,selected,true).order);
    for(const e of graph.outgoing.get(selected)||[])related.add(e.to);
    for(const [id,button]of buttons){button.setAttribute('aria-pressed',id===selected);button.classList.toggle('map-related',related.has(id));button.classList.toggle('map-match',matches.has(id));}
    for(const {edge,path}of paths)path.classList.toggle('map-related',edge.from===selected||edge.to===selected||related.has(edge.from)&&related.has(edge.to));
    selectedPreview();drawMini();
    status.textContent=`${layout.positions.size.toLocaleString()} nodes · ${edges.length.toLocaleString()} connections · ${scope==='all'?'the entire indexed map':'the complete selected ancestry'} · no count or depth limits`;
  }

  function draw() {
    for(const button of buttons.values())button.remove();buttons.clear();
    world.querySelectorAll('.map-group-heading').forEach(n=>n.remove());
    svg.querySelectorAll('.map-edge').forEach(n=>n.remove());paths.length=0;
    world.style.width=layout.width+'px';world.style.height=layout.height+'px';
    svg.setAttribute('width',layout.width);svg.setAttribute('height',layout.height);svg.setAttribute('viewBox',`0 0 ${layout.width} ${layout.height}`);
    const edgeFragment=document.createDocumentFragment();
    edges.forEach((edge,index)=>{
      const routed=routeEdge(edge,layout,index);if(!routed)return;
      const path=document.createElementNS('http://www.w3.org/2000/svg','path');
      const context=['model assumption','numerical method','source definition','model context'].includes(edge.kind);
      path.setAttribute('class',`map-edge ${context?'context-edge':physicsEdge(edge)?'quantity-edge':'code-edge'}`);
      path.setAttribute('d',routed.path);path.setAttribute('marker-end','url(#variable-map-arrow)');
      path.dataset.from=edge.from;path.dataset.to=edge.to;path.dataset.kind=edge.kind;
      const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=`${edge.kind} · ${edge.from} → ${edge.to}`;path.append(title);edgeFragment.append(path);paths.push({edge,path});
    });svg.append(edgeFragment);
    const nodeFragment=document.createDocumentFragment();
    for(const group of layout.groups){const heading=el('button',{type:'button',class:'map-group-heading'},`${group.name} · ${group.count}`);heading.style.cssText=`left:${group.x}px;top:${group.y}px;width:${group.width}px`;heading.onclick=()=>fitBounds(group);nodeFragment.append(heading);}
    for(const [id,p]of layout.positions) {
      const n=graph.nodes.get(id),button=el('button',{type:'button',class:`map-node ${['model assumption','numerical method'].includes(n.family)?'assumption-node':n.physics?'physics-node-card':'code-node-card'}`,'data-node-id':id,'aria-pressed':id===selected,title:`${n.name}\n${n.id}${n.unit?' · '+n.unit:''}`});
      button.style.cssText=`left:${p.x}px;top:${p.y}px;width:${p.width}px;min-height:${p.height}px`;
      const top=el('span',{class:'map-node-top'});if(n.symbol)top.append(math(n.symbol));top.append(el('span',{class:'kind-badge'},n.family));button.append(top,el('strong',{},graphLabel(n)));
      const bottom=el('span',{class:'map-node-bottom'});if(n.unit)bottom.append(unit(n.unit));if(n.module)bottom.append(el('span',{},n.module));if(n.provenance)bottom.append(el('span',{},n.provenance));button.append(bottom);
      if(['leaf','setting','derived'].includes(n.type)&&n.value)button.append(el('span',{class:'map-node-value'},n.value));
      button.onclick=()=>onSelect(id);buttons.set(id,button);nodeFragment.append(button);
    }
    world.append(nodeFragment);updateHighlights();
    all.setAttribute('aria-pressed',scope==='all');ancestry.setAttribute('aria-pressed',scope==='ancestry');
    root.dataset.nodeCount=layout.positions.size;root.dataset.edgeCount=edges.length;
    if(viewport.clientWidth)fitMap();else pendingFit=true;
  }

  function rebuild() {
    if(scope==='all'){layout=universe;edges=allEdges;}
    else {
      const trace=traceGraph(graph,selected,true),ids=new Set(trace.order);edges=[];
      for(const id of trace.order)for(const edge of inputsFor(graph,id,true))if(ids.has(edge.from))edges.push({...edge,to:id});
      layout=layoutDerivation(graph,trace.order,edges);
    }
    draw();
  }
  const reveal=()=>root.scrollIntoView({block:'start'});
  all.onclick=()=>{scope='all';rebuild();reveal();};ancestry.onclick=()=>{scope='ancestry';rebuild();focusNode(selected);reveal();};
  fit.onclick=()=>{fitMap();reveal();};focus.onclick=()=>{focusNode(selected);reveal();};
  plus.onclick=()=>zoom(1.5);minus.onclick=()=>zoom(1/1.5);
  viewport.addEventListener('wheel',event=>{event.preventDefault();const r=viewport.getBoundingClientRect();zoom(Math.exp(-event.deltaY*.002),event.clientX-r.x,event.clientY-r.y);},{passive:false});
  viewport.onpointerdown=event=>{if(event.target.closest('button')||event.target===mini)return;drag={x:event.clientX,y:event.clientY,camera:{...camera}};viewport.setPointerCapture(event.pointerId);viewport.classList.add('dragging');};
  viewport.onpointermove=event=>{if(!drag)return;camera={...drag.camera,x:drag.camera.x+event.clientX-drag.x,y:drag.camera.y+event.clientY-drag.y};cameraChanged();};
  viewport.onpointerup=viewport.onpointercancel=()=>{drag=null;viewport.classList.remove('dragging');};
  viewport.onkeydown=event=>{
    if(event.target!==viewport)return;
    if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','=','0'].includes(event.key))event.preventDefault();
    if(event.key==='+'||event.key==='=')zoom(1.5);else if(event.key==='-')zoom(1/1.5);else if(event.key==='0')fitMap();
    else{const delta={ArrowLeft:[80,0],ArrowRight:[-80,0],ArrowUp:[0,80],ArrowDown:[0,-80]}[event.key];if(delta){camera.x+=delta[0];camera.y+=delta[1];cameraChanged();}}
  };
  mini.onclick=event=>{const r=mini.getBoundingClientRect(),ratio=Number(mini.dataset.ratio),pad=Number(mini.dataset.padding);const x=((event.clientX-r.x)*mini.width/r.width-pad)/ratio,y=((event.clientY-r.y)*mini.height/r.height-pad)/ratio;camera.x=viewport.clientWidth/2-x*camera.scale;camera.y=viewport.clientHeight/2-y*camera.scale;cameraChanged();};
  new ResizeObserver(()=>{
    const width=viewport.clientWidth,height=viewport.clientHeight;if(!width||!height)return;
    if(pendingFit){pendingFit=false;fitMap();}
    else{if(previousSize){camera.x+=(width-previousSize.width)/2;camera.y+=(height-previousSize.height)/2;}cameraChanged();}
    previousSize={width,height};
  }).observe(viewport);
  draw();
  return {element:root,select(id,{focus=false}={}){selected=id;if(scope==='ancestry')rebuild();else updateHighlights();if(focus)focusNode(id);},
    highlight(ids){matches=new Set(ids);for(const [id,button]of buttons)button.classList.toggle('map-match',matches.has(id));},fit:fitMap};
}
