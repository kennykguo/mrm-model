import {CONCEPTS, LINKS, TOPICS, CONTEXTS, isPhysics, topic, normalizeTex, quantity, withNotation, recordKind} from './physics.js';
import {createExplorer} from './explorer.js';

const $ = id => document.getElementById(id);
const nodes = new Map(), incoming = new Map(), outgoing = new Map();
let data, activeConcept = 'temperature', activeTopic = 'all';
let explorer;
const texCache = new Map();
const FRIENDLY = {
  'ring_float.t_ring':'ring temperature', 'ring_float.t_new':'temperature solved with optical heating',
  'ring_float.he_p':'heater power', 'ring_float.p_abs':'absorbed optical power', 'ring_float.det':'resonance minus laser wavelength',
  'P.th_ring_obs':'ring temperature observer index', 'S.th_t_obs':'thermal observer temperatures',
  'S.th_x':'thermal temperature-rise states', 'obs.tx.monitor.latest':'released monitor ADC code',
  'selfheat.beta':'relative thermal tuning curvature', 'selfheat.s0':'thermal tuning slope at the reference temperature',
};

function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [key,value] of Object.entries(attrs)) e.setAttribute(key,value);
  if (text !== undefined) e.textContent = text;
  return e;
}
function math(tex, display = false, fallback = '') {
  const result = el('span', {class:display ? 'math formula' : 'math'});
  const normalized = normalizeTex(tex);
  if (window.katex) {
    const key = `${display}:${normalized}`;
    try {
      if (!texCache.has(key)) texCache.set(key, katex.renderToString(String.raw`\mathtt{` + normalized.replace(/\\mathrm\b/g, String.raw`\mathtt`) + '}', {displayMode:display,throwOnError:true,trust:false,output:'htmlAndMathml'}));
      result.innerHTML = texCache.get(key);
    } catch (error) {
      result.textContent = fallback || normalized;
      result.classList.add('math-error');
      console.error('Invalid reference notation', normalized, error.message);
    }
  } else result.textContent = fallback || normalized;
  return result;
}
function unit(text) {
  if (!text) return el('span',{class:'note'},'not specified');
  if (text==='degC') return math(String.raw`{}^{\circ}\mathrm C`);
  if (text==='ohm' || text===String.raw`\Omega`) return math(String.raw`\Omega`);
  if (text==='m/V^m_j') return math(String.raw`\mathrm m/\mathrm V^{m_j}`);
  if (String(text).includes('sqrt(Hz)')) return math(String.raw`\mathrm{`+String(text).split('/')[0]+String.raw`}/\sqrt{\mathrm{Hz}}`);
  const escaped = String(text).replace(/\\/g,'').replace(/([{}_#%&$])/g,'\\$1');
  return math(String.raw`\mathrm{` + escaped.replace(/Ω/g,String.raw`\Omega`).replace(/\bu(?=[Am])/g,String.raw`\mu `) + '}', false, text);
}
function title(n) {
  if (FRIENDLY[n.id]) return FRIENDLY[n.id];
  if (!isPhysics(n)) return n.label || n.id;
  return quantity(n).name;
}
function context(n) { return CONTEXTS[n.type] || n.type; }
function badge(n) { return el('span',{class:'kind-badge'},recordKind(n)); }

function buildMap() {
  const readingOrder=['heater','temperature','resonance','phase','loss','absorption','drop','current'];
  for (const id of readingOrder) {
    const c=CONCEPTS.find(c=>c.id===id);
    const b = el('button',{type:'button',class:'physics-node',id:`quantity-${c.id}`,'aria-pressed':c.id===activeConcept,'aria-controls':'concept'});
    b.style.gridColumn = c.position[0]+1;
    b.style.gridRow = c.position[1]+1;
    b.append(math(c.symbol),el('strong',{},c.name),el('span',{class:'node-unit'},c.unit));
    b.onclick = () => selectConcept(c.id);
    $('physics-map').append(b);
  }
  new ResizeObserver(drawLinks).observe($('physics-map'));
  document.fonts?.ready.then(drawLinks);
}
function drawLinks() {
  const root = $('physics-map'), svg = $('physics-links');
  if (!root.clientWidth || getComputedStyle(root).gridTemplateColumns.split(' ').length < 3) return;
  svg.querySelectorAll('path.connection').forEach(e => e.remove());
  const box = root.getBoundingClientRect();
  svg.setAttribute('viewBox',`0 0 ${box.width} ${box.height}`);
  const anchor = (id,side) => {
    const b = $(`quantity-${id}`).getBoundingClientRect();
    return {x:b.x-box.x+(side==='left' ? -5 : side==='right' ? b.width+5 : b.width/2),
      y:b.y-box.y+(side==='top' ? -5 : side==='bottom' ? b.height+5 : b.height/2)};
  };
  for (const link of LINKS) {
    const a=anchor(link.from,link.ports[0]), b=anchor(link.to,link.ports[1]);
    const d=link.feedback ? `M ${a.x} ${a.y} V ${(a.y+b.y)/2} H ${b.x} V ${b.y}` : `M ${a.x} ${a.y} L ${b.x} ${b.y}`;
    const path=document.createElementNS('http://www.w3.org/2000/svg','path');
    path.setAttribute('d',d);path.setAttribute('class',`connection${link.feedback ? ' feedback' : ''}`);
    path.setAttribute('marker-end','url(#arrow)');svg.append(path);
  }
}
function selectConcept(id, save = true) {
  const c=CONCEPTS.find(x => x.id===id) || CONCEPTS[1];activeConcept=c.id;
  for (const b of $('physics-map').querySelectorAll('button')) b.setAttribute('aria-pressed',b.id===`quantity-${c.id}`);
  const panel=$('concept');
  panel.replaceChildren(el('p',{class:'eyebrow'},'selected physical quantity'),el('h3',{},c.question[0].toLowerCase()+c.question.slice(1)),el('p',{},c.meaning),math(c.equation,true));
  if (c.extra) panel.append(math(c.extra,true));
  if (c.extra2) panel.append(math(c.extra2,true));
  const definitions=el('dl',{class:'definitions'});
  for (const [symbol,meaning,units] of c.definitions) {
    const dd=el('dd');dd.append(el('span',{},meaning),unit(units));
    definitions.append(el('dt'),dd);definitions.lastElementChild.previousElementSibling.append(math(symbol));
  }
  panel.append(definitions,el('p',{class:'note'},c.note));
  const details=el('details');details.append(el('summary',{},'how does this appear in the code?'),el('p',{class:'code-source'},c.source));
  if (c.id==='phase') details.append(el('p',{class:'note'},'the stored phase rows are reference phases at each data level. the trial-temperature calculation adds the temperature-induced phase change.'));
  const related=c.records.filter(id => nodes.has(id));
  if (related.length) {
    details.append(el('p',{class:'note'},'related stored values'));
    for (const id of related) {
      const b=el('button',{type:'button',class:'code-link'},id);b.onclick=()=>explorer.select(id,true,true);details.append(b);
    }
  }
  panel.append(details);
  if (save) history.replaceState(null,'',`#physics/${id}`);
}
function setMode(mode, save = true) {
  const physical=mode==='physics';
  for(const key of ['derive','physics','model']){$(key).hidden=key!==mode;$('mode-'+key).setAttribute('aria-pressed',key===mode);}
  $('mode-note').textContent=physical ? 'physical quantities first' : mode==='derive'?'definitions from the ground up':'storage, algorithms, timing, and interfaces';
  if (physical) requestAnimationFrame(drawLinks);
  if (save) history.replaceState(null,'',physical ? `#physics/${activeConcept}` : mode==='derive'?`#derive/${encodeURIComponent(explorer?.selected()||'equation.backscatter_rate')}`:'#model');
}
function setupSymbols() {
  for (const [key,text] of [['all','all physics'],...Object.entries(TOPICS)]) {
    const b=el('button',{type:'button','aria-pressed':key===activeTopic},text);
    b.onclick=()=>{ activeTopic=key;for(const x of $('topics').children)x.setAttribute('aria-pressed',x===b);renderSymbols(); };
    $('topics').append(b);
  }
  $('symbol-search').oninput=renderSymbols;
  renderSymbols();
}
function match(n, query) {
  const words=query.toLowerCase().split(/\s+/).filter(Boolean);
  const haystack=[n.id,n.label,n.meaning,n.symbol,n.symbol_text,normalizeTex(n.symbol),quantity(n).name,FRIENDLY[n.id]].join(' ').toLowerCase();
  return words.every(word => haystack.includes(word));
}
function renderSymbols() {
  const all=data.nodes.filter(isPhysics);
  const found=all.filter(n=>(activeTopic==='all'||topic(n)===activeTopic)&&match(n,$('symbol-search').value));
  found.sort((a,b)=>title(a).localeCompare(title(b)) || a.layer-b.layer);
  $('symbol-count').textContent=`${found.length} matching quantities · ${all.length} physical quantities with defined symbols in this reference`;
  const list=$('symbol-list');list.replaceChildren();
  const head=el('div',{class:'symbol-head','aria-hidden':'true'});
  head.append(el('span',{},'symbol'),el('span',{},'meaning & context'),el('span',{},'units'));list.append(head);
  for (const n of found) {
    const row=el('div',{class:'symbol-row'}), meaning=el('div',{class:'symbol-meaning'});
    const q=quantity(n);
    meaning.append(el('strong',{},q.name));
    if(!q.name)meaning.append(el('p',{},q.meaning));
    meaning.append(el('span',{class:'note'},context(n)));
    const code=el('details');code.append(el('summary',{},'code mapping & recorded context'),el('code',{},n.id));
    if(n.meaning)code.append(el('p',{class:'note'},n.meaning));
    const b=el('button',{class:'text-button',type:'button'},'trace all inputs →');b.onclick=()=>explorer.select(n.id,true,true);code.append(b);meaning.append(code);
    row.append(math(n.symbol,false,n.symbol_text),meaning,unit(q.unit));list.append(row);
  }
  if (!found.length) list.append(el('p',{class:'empty'},'no matching physical quantities. try another symbol or topic.'));
}
function valueCard(id) {
  const n=nodes.get(id), b=el('button',{type:'button',class:'value-card'});
  const line=el('span',{class:'value-top'});line.append(badge(n));
  if (n.symbol) line.append(math(n.symbol,false,n.symbol_text));
  b.append(line,el('strong',{},title(n)),el('code',{},id));
  if (n.meaning) b.append(el('span',{class:'value-meaning'},isPhysics(n)?quantity(n).meaning:n.meaning));
  if (!n.symbol) b.append(el('span',{class:'note'},'no equation symbol is assigned to this record'));
  b.onclick=()=>showRecord(id);return b;
}
function renderModel() {
  const type=$('model-kind').value, layer=$('model-layer').value;
  const found=data.nodes.filter(n=>(type==='all'||isPhysics(n)===(type==='physics'))&&(layer==='all'||n.layer===Number(layer))&&match(n,$('model-search').value));
  found.sort((a,b)=>a.layer-b.layer || a.id.localeCompare(b.id));
  $('model-count').textContent=`${found.length} matching records · ${data.nodes.length} stored values in the generated model`;
  $('model-list').replaceChildren(...found.map(n=>valueCard(n.id)));
  if (!found.length) $('model-list').append(el('p',{class:'empty'},'no matching stored values. try another name or category.'));
}
function resetModel() {
  $('model-focus').hidden=true;$('model-browse').hidden=false;renderModel();
}
function dependencyLane(ids, heading) {
  const section=el('section',{class:'dependency-lane'});section.append(el('h4',{},heading));
  const sorted=[...new Set(ids)].sort((a,b)=>a.localeCompare(b));
  for (const id of sorted) section.append(valueCard(id));
  if (!sorted.length) section.append(el('p',{class:'empty'},'no immediate dependencies recorded'));
  return section;
}
function showRecord(id, save = true) {
  const n=nodes.get(id);if (!n)return;
  setMode('model',false);$('model-browse').hidden=true;$('model-focus').hidden=false;
  $('focus-title').textContent=title(n);
  const selected=el('section',{class:'selected-record'});selected.append(el('p',{class:'eyebrow'},'selected record'),badge(n));
  if(n.symbol)selected.append(math(n.symbol,false,n.symbol_text));
  selected.append(el('strong',{},title(n)),el('code',{},id));
  $('dependency-map').replaceChildren(dependencyLane(incoming.get(id),'computed from →'),selected,dependencyLane(outgoing.get(id),'→ used by'));
  const record=$('record');record.replaceChildren(el('h3',{},'what does this stored value mean?'),el('p',{},isPhysics(n)?quantity(n).meaning:n.meaning||title(n)));
  if(n.formula_tex)record.append(math(n.formula_tex,true));
  else if(n.formula)record.append(el('p',{class:'code-source'},n.formula));
  if(n.note)record.append(el('p',{class:'note'},n.note));
  const detail=el('dl',{class:'record-details'});
  for (const [key,value] of [['category',data.layers.find(x=>x.id===n.layer)?.title],['units',isPhysics(n)?quantity(n).unit:n.unit],['array shape',n.shape],['preset value',n.value],['storage',n.storage],['updated by',n.updated_by],['initial state',n.init],['source',n.source],['provenance',n.provenance],['relation',n.relation],['textbook equation',n.eq]]) {
    if(value===undefined||value===null||value==='')continue;
    detail.append(el('dt',{},key),el('dd',{},Array.isArray(value)?value.join(', '):String(value)));
  }
  record.append(detail);
  const trace=el('button',{type:'button'},'trace all inputs from the foundations →');trace.onclick=()=>explorer.select(n.id,true,true);record.append(trace);
  if(n.other_sources?.length)record.append(el('p',{class:'note'},'other inputs · '+n.other_sources.join(', ')));
  if(n.absent_sources?.length)record.append(el('p',{class:'note'},'not present in this preset · '+n.absent_sources.join(', ')));
  if(n.draws?.length)record.append(el('p',{class:'note'},'uncertainty and variation · '+n.draws.join('; ')));
  if(save)history.replaceState(null,'',`#${encodeURIComponent(id)}`);
  $('model').scrollIntoView({block:'start'});
}
function route() {
  let hash='';try{hash=decodeURIComponent(location.hash.slice(1));}catch{}
  if(hash.startsWith('derive/')){setMode('derive',false);explorer?.select(hash.slice(7),false);}
  else if(nodes.has(hash))showRecord(hash,false);
  else if(hash==='model'){setMode('model',false);resetModel();}
  else if(hash.startsWith('physics/')){setMode('physics',false);selectConcept(hash.slice(8),false);}
  else setMode('derive',false);
}
async function init() {
  try {
    const response=await fetch('data.json');if(!response.ok)throw Error(`data.json returned ${response.status}`);
    data=await response.json();
    data.nodes=data.nodes.map(withNotation);
    for(const n of data.nodes){nodes.set(n.id,n);incoming.set(n.id,[]);outgoing.set(n.id,[]);}
    for(const [from,to] of data.edges){incoming.get(to).push(from);outgoing.get(from).push(to);}
    for(const layer of data.layers)$('model-layer').append(el('option',{value:layer.id},layer.title.toLowerCase()));
    $('jump-symbols').onclick=()=>{setMode('physics');$('symbols-title').scrollIntoView({block:'start'});};
    $('mode-derive').onclick=()=>setMode('derive');$('mode-physics').onclick=()=>setMode('physics');$('mode-model').onclick=()=>setMode('model');
    $('model-search').oninput=resetModel;$('model-kind').onchange=resetModel;$('model-layer').onchange=resetModel;
    $('back-model').onclick=()=>{resetModel();history.replaceState(null,'','#model');};
    buildMap();setupSymbols();renderModel();selectConcept(activeConcept,false);
    explorer=await createExplorer(data,{el,math,unit,setMode});$('loading').hidden=true;route();
    window.addEventListener('hashchange',route);
    if(!window.katex){const notice=el('p',{class:'note'},'math rendering could not load. the reference still shows the available text notation.');$('content').prepend(notice);}
  } catch(error) {
    $('loading').hidden=true;$('error').hidden=false;
    $('error').textContent='the variable reference could not load. open this page through the local server, then reload.';
    console.error(error);
  }
}
init();
