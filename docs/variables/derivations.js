import {isPhysics, quantity, normalizeTex, withNotation} from './physics.js';
import {MODEL_CONTEXTS} from './model-context.js';

const raw = String.raw;
export const BACKSCATTER_TARGET = 'equation.backscatter_rate';
const fn = name => `fn:mrm.plant.optics.${name}`;
const cfg = name => `cfg.${name}`;
const CODE_LABELS={'P.th_ring_obs':'ring temperature observer index','obs.tx.monitor.latest':'released monitor ADC code'};

// These equations follow the nominal preset path; equivalent identities are displayed separately.
export const DEFINITIONS = [
  {id:'constant.pi',name:'circle constant',symbol:raw`\pi`,unit:'1',family:'mathematical constant',value:'3.141592653589793',meaning:'the ratio of a circle’s circumference to its diameter',aliases:['pi','math.pi'],source_ids:['const:mrm.plant.optics.TWO_PI'],inputs:[]},
  {id:'constant.c',name:'speed of light in vacuum',symbol:'c',unit:'m/s',family:'physical constant',value:'299792458',meaning:'the exact SI speed of light in vacuum',aliases:['c','c0','C0'],source_ids:['const:mrm.units.C0'],inputs:[]},
  {id:'constant.qe',name:'elementary charge',symbol:'q_e',unit:'C',family:'physical constant',value:'1.602176634e-19',meaning:'the exact SI elementary charge',aliases:['q_e','Q_E'],source_ids:['const:mrm.units.Q_E'],inputs:[]},
  {id:'constant.h',name:'Planck constant',symbol:'h',unit:'J s',family:'physical constant',value:'6.62607015e-34',meaning:'the exact SI Planck constant',aliases:['h','H_PLANCK'],source_ids:['const:mrm.units.H_PLANCK'],inputs:[]},
  {id:'constant.kb',name:'Boltzmann constant',symbol:'k_B',unit:'J/K',family:'physical constant',value:'1.380649e-23',meaning:'the exact SI Boltzmann constant',aliases:['k_B','K_B'],source_ids:['const:mrm.units.K_B'],inputs:[]},
  {id:'constant.eps0',name:'vacuum permittivity',symbol:raw`\epsilon_0`,unit:'F/m',family:'physical constant',value:'8.8541878128e-12',meaning:'the vacuum permittivity used by the model',aliases:['epsilon_0','EPS0'],source_ids:['const:mrm.units.EPS0'],inputs:[]},
  {id:'equation.fsr_si',name:'free spectral range in hertz',symbol:raw`\Delta f_{\mathrm{FSR}}`,unit:'Hz',family:'equation',
    formula_tex:raw`\Delta f_{\mathrm{FSR}}=(10^{12}\,\mathrm{Hz/THz})\,\Delta f_{\mathrm{FSR,THz}}`,
    equivalent:raw`\Delta f_{\mathrm{FSR}}=\frac{c}{n_g L}`,meaning:'convert the preset FSR to SI. the preset supplies the FSR and group index, then derives the radius; the geometry identity is the same constraint read in another direction.',
    aliases:['FSR','fsr_hz','free spectral range'],inputs:[cfg('ring.optics.fsr_thz')],source_ids:['fn:mrm.params.relations._solve_r3_geometry'],records:['cfg.derived.optics.fsr_hz','pp.optics.fsr_hz']},
  {id:'equation.splitting_si',name:'configured mode splitting in hertz',symbol:raw`\Delta f_{\mathrm{split}}`,unit:'Hz',family:'equation',
    formula_tex:raw`\Delta f_{\mathrm{split}}=(10^9\,\mathrm{Hz/GHz})\,\Delta f_{\mathrm{split,GHz}}`,
    meaning:'convert the configured splitting anchor from GHz. disabling backscatter suppresses the realized coupling even when this configured anchor is nonzero.',
    aliases:['splitting','mode splitting','backscatter_splitting_hz'],inputs:[cfg('ring.optics.backscatter_splitting_ghz')],source_ids:['fn:mrm.params.realize._backscatter_rb']},
  {id:'equation.beta_nominal',name:'nominal complex coupling in cycles per second',symbol:raw`\beta_n`,unit:'Hz',family:'equation',
    formula_tex:raw`\beta_n=\begin{cases}0&\text{backscatter off}\\\Delta f_{\mathrm{split}}/2&\text{backscatter on, variation off}\end{cases}`,
    meaning:'when backscatter is enabled and variation is off, each resonance order has the same real half-splitting. disabling backscatter gives an effective zero coupling. with variation on, the code draws complex coefficients with the stated RMS magnitude instead.',
    aliases:['beta','beta_hz','half_splitting_hz','beta_n'],inputs:['equation.splitting_si','cfg.fidelity.backscatter'],source_ids:['fn:mrm.params.realize._backscatter_rb'],scope:'the nonzero numerical example explicitly enables backscatter and disables device variation; n identifies the resonance order'},
  {id:'equation.round_trip_backscatter',name:'complex coupling per round trip',symbol:raw`r_{b,n}`,unit:'1',family:'equation',
    formula_tex:raw`r_{b,n}=\frac{2\pi\beta_n}{\Delta f_{\mathrm{FSR}}}`,
    meaning:'convert the complex coupling rate in Hz to a dimensionless coupling per ring trip',
    aliases:['r_b','rb','rb_n','round trip backscatter','backscatter_rb'],inputs:['constant.pi','equation.beta_nominal','equation.fsr_si'],source_ids:[fn('backscatter_hz_to_round_trip_coupling'),'fn:mrm.params.realize._backscatter_rb'],records:['pp.optics.backscatter_rb']},
  {id:'equation.round_trip_delay',name:'round-trip group delay',symbol:raw`t_{\mathrm{rt}}`,unit:'s',family:'equation',
    formula_tex:raw`t_{\mathrm{rt}}=\frac{n_g L}{c}`,
    equivalent:raw`t_{\mathrm{rt}}=\tau_{\mathrm{rt}}=T_{\mathrm{rt}}=\frac{1}{\Delta f_{\mathrm{FSR}}}`,
    meaning:'divide the ring length by group velocity c/n_g. this also equals the inverse FSR. the preset supplies FSR and group index, then derives the radius and length; the code and reference use several names for the resulting delay.',
    aliases:['t_rt','t_rt_s','T_rt','tau_rt','round_trip_time_s','round trip time'],inputs:['cfg.ring.optics.n_g','cfg.derived.optics.length_m','constant.c'],source_ids:['fn:mrm.params.relations._solve_r3_geometry'],records:['cfg.derived.optics.round_trip_time_s','pp.optics.round_trip_time_s']},
  {id:BACKSCATTER_TARGET,name:'TCMT complex backscatter rate',symbol:'g',unit:'rad/s',family:'equation',
    formula_tex:raw`g_n=\frac{r_{b,n}}{t_{\mathrm{rt}}}`,
    equivalent:raw`g_n=2\pi\beta_n`,
    meaning:'divide coupling per trip by seconds per trip to get the continuous-time angular coupling rate. the complex phase is retained; this is a coupling amplitude, not a probability or a power fraction.',
    aliases:['g','g_n','g=rb/trt',raw`g=\frac{r_b}{t_\mathrm{rt}}`,'tcmt_backscatter_rate','backscatter_rate'],inputs:['equation.round_trip_backscatter','equation.round_trip_delay'],source_ids:[fn('tcmt_backscatter_rate')],scope:'the helper defines the conversion. the archived steady-state TCMT comparison also performs this division directly. the exact scattering model uses the round-trip coefficient instead.'},
  {id:'equation.energy_decay_intrinsic',name:'intrinsic energy decay rate',symbol:raw`\kappa_i`,unit:'1/s',family:'equation',formula_tex:raw`\kappa_i=-\frac{2\ln a}{t_{\mathrm{rt}}}`,meaning:'field retention a becomes energy retention a squared; equate it to exponential energy decay over one round trip',aliases:['kappa_i','intrinsic_decay_rate'],inputs:['cfg.derived.optics.round_trip_amplitude','equation.round_trip_delay'],source_ids:[fn('energy_decay_rates_from_ring'),fn('exact_to_tcmt')]},
  {id:'equation.energy_decay_through',name:'energy leakage rate to the bus',symbol:raw`\kappa_e`,unit:'1/s',family:'equation',formula_tex:raw`\kappa_e=-\frac{\ln(1-\kappa_t^2)}{t_{\mathrm{rt}}}`,meaning:'convert the energy retained after the through coupler to exponential energy decay per second',aliases:['kappa_e','through_coupling_decay_rate'],inputs:['cfg.derived.optics.kappa2_through','equation.round_trip_delay'],source_ids:[fn('energy_decay_rates_from_ring'),fn('exact_to_tcmt')],scope:'the symbol kappa with subscript t squared is a dimensionless power fraction; kappa with subscript e is a rate'},
  {id:'equation.energy_decay_drop',name:'energy leakage rate to the drop guide',symbol:raw`\kappa_d^{(\mathrm{rate})}`,unit:'1/s',family:'equation',formula_tex:raw`\kappa_d^{(\mathrm{rate})}=-\frac{\ln(1-\kappa_d^2)}{t_{\mathrm{rt}}}`,meaning:'convert the energy retained after the drop coupler to exponential energy decay per second',aliases:['kappa_d','drop_coupling_decay_rate'],inputs:['cfg.derived.optics.kappa2_drop','equation.round_trip_delay'],source_ids:[fn('energy_decay_rates_from_ring'),fn('exact_to_tcmt')],scope:'the rate superscript distinguishes the TCMT decay rate from the dimensionless coupler power fraction'},
];

export function searchKey(value) {
  return normalizeTex(value).toLowerCase().replace(/\\(?:mathrm|mathtt|text|frac|left|right|overline|bar|mathcal)\b/g,' ').replace(/\\/g,'').replace(/[^\p{L}\p{N}]+/gu,' ').trim();
}
const compact = value => searchKey(value).replace(/\s/g,'');

export function buildGraph(data, sourceIndex) {
  const nodes=new Map(), edges=[];
  for(const original of data.nodes.map(withNotation)) {
    const q=quantity(original), physical=isPhysics(original);
    nodes.set(original.id,{...original,name:physical?q.name:CODE_LABELS[original.id]||original.label||original.id,
      unit:physical?q.unit:original.unit, family:physical?'physical quantity':'model / simulation',
      physics:physical, aliases:[original.id,original.path,original.symbol_text,original.label].filter(Boolean),
      source_ids:sourceIndex.record_sources?.[original.id]||[],source_ref:sourceIndex.record_locations?.[original.id],
      boundary:original.type==='leaf'?'preset anchor':original.type==='setting'?'preset setting':'recorded input boundary'});
  }
  for(const n of sourceIndex.nodes)nodes.set(n.id,{...n,physics:n.physics||false,aliases:[...n.aliases||[],n.name,n.qualified,n.signature].filter(Boolean),boundary:n.family==='function input'?'caller-supplied physical argument':n.family==='function'?'function argument or external call':'source-defined value'});
  for(const [from,to] of data.edges)edges.push({from,to,kind:'recorded input'});
  for(const [from,to,kind] of sourceIndex.edges)edges.push({from,to,kind});
  for(const definition of [...DEFINITIONS,...MODEL_CONTEXTS])nodes.set(definition.id,{...definition,physics:true,boundary:definition.family==='model assumption'?'stated compact-model assumption':definition.family==='numerical method'?'chosen numerical method':'physical definition'});

  // Resolve source functions by exact terminal name when the module has a private implementation name.
  for(const definition of [...DEFINITIONS,...MODEL_CONTEXTS]) {
    const n=nodes.get(definition.id);
    n.source_ids=(definition.source_ids||[]).map(id=>{
      if(nodes.has(id))return id;
      const name=id.split('.').pop().replace(/^_/,'');
      return [...nodes.keys()].find(k=>k.startsWith('fn:mrm.params.relations.') && k.split('.').pop().replace(/^_/,'').toLowerCase()===name.toLowerCase()) || id;
    }).filter(id=>nodes.has(id));
    for(const from of definition.inputs)if(nodes.has(from))edges.push({from,to:definition.id,kind:'equation input'});
    for(const record of definition.records||[])if(nodes.has(record))edges.push({from:definition.id,to:record,kind:'defined by'});
    for(const consumer of definition.applies_to||[])if(nodes.has(consumer))edges.push({from:definition.id,to:consumer,kind:definition.family==='numerical method'?'numerical method':'model assumption'});
  }
  const setting=path=>String(nodes.get('cfg.'+path)?.value||'not recorded');
  const backscatter=nodes.get(BACKSCATTER_TARGET);
  backscatter.scope+=' indexed reference settings — optics '+setting('fidelity.optics_model')+', backscatter '+setting('fidelity.backscatter')+'. the nonzero numerical example explicitly changes backscatter to on.';
  if(setting('fidelity.backscatter')==='off')backscatter.reference_formula=raw`r_{b,n}=0\quad\Longrightarrow\quad g_n=0`;
  nodes.get('equation.round_trip_delay').scope='the inverse-FSR identity uses the local group-delay spacing of relation R3. finite spacing between adjacent resonances can vary with dispersion.';

  // Keep the actual preset derivation path; link constants when the stored equation names them.
  for(const n of nodes.values())if(n.formula_tex && n.id.startsWith('cfg.')) {
    if(/(?:\bc\b)/.test(normalizeTex(n.formula_tex)))edges.push({from:'constant.c',to:n.id,kind:'equation constant'});
    if(normalizeTex(n.formula_tex).includes('\\pi'))edges.push({from:'constant.pi',to:n.id,kind:'equation constant'});
  }
  for(const n of nodes.values())if(n.family==='function output'&&n.formula_tex) {
    if(/\bc\b/.test(n.formula_tex))edges.push({from:'constant.c',to:n.id,kind:'equation constant'});
    if(n.formula_tex.includes('\\pi'))edges.push({from:'constant.pi',to:n.id,kind:'equation constant'});
  }
  const incoming=new Map([...nodes.keys()].map(id=>[id,[]])),outgoing=new Map([...nodes.keys()].map(id=>[id,[]]));
  for(const edge of edges)if(nodes.has(edge.from)&&nodes.has(edge.to)){incoming.get(edge.to).push(edge);outgoing.get(edge.from).push(edge);}
  for(const n of nodes.values()) {
    n.search_aliases=[n.id,n.symbol,n.symbol_text,...n.aliases||[]].filter(Boolean).map(compact);
    n.search_text=searchKey([n.id,n.name,n.meaning,n.formula_tex,n.formula,n.equivalent,n.signature,n.source_ref?.code,...n.comments||[],...n.aliases||[]].join(' '));
  }
  return {nodes,incoming,outgoing,meta:sourceIndex.meta};
}

export function searchGraph(graph, query, category='all') {
  const rawQuery=query.trim(), key=searchKey(rawQuery), full=compact(rawQuery), words=key.split(' ').filter(Boolean);
  const matches=[];
  for(const n of graph.nodes.values()) {
    if(category==='physics'&&!n.physics || category==='implementation'&&n.physics)continue;
    const exact=n.search_aliases.some(alias=>alias===full);
    const contains=words.every(word=>n.search_text.includes(word));
    if(rawQuery && !exact && (!contains || full.length<2))continue;
    let score=exact?100:0;
    if(n.family==='equation')score+=25;
    if(n.physics)score+=10;
    if(compact(n.name).startsWith(full))score+=5;
    matches.push({node:n,score});
  }
  return matches.sort((a,b)=>b.score-a.score||a.node.name.localeCompare(b.node.name)).map(x=>x.node);
}

export function inputsFor(graph,id,includeImplementation=false) {
  const allowed=new Set(['recorded input','equation input','equation constant','defined by','function input','model assumption','numerical method']);
  const node=graph.nodes.get(id);
  const edges=graph.incoming.get(id)||[];
  const valid=edges.filter(e=>allowed.has(e.kind)||(node?.family==='function'&&['call','source reference','source read','code constant'].includes(e.kind))||(node?.family==='code constant'&&e.kind==='code constant'));
  const result=new Map();
  const visit=(edge,via,seen)=>{
    if(seen.has(edge.from))return;
    const n=graph.nodes.get(edge.from);if(!n)return;
    if(includeImplementation || n.physics || node?.family==='function'||node?.family==='code constant') {
      if(!result.has(n.id))result.set(n.id,{...edge,via});return;
    }
    const upstream=(graph.incoming.get(n.id)||[]).filter(e=>allowed.has(e.kind));
    if(!upstream.length){if(!result.has(n.id))result.set(n.id,{...edge,via,boundary:true});return;}
    for(const e of upstream)visit(e,[...via,n.id],new Set([...seen,n.id]));
  };
  for(const edge of valid)visit(edge,[],new Set([id]));
  return [...result.values()];
}

export function traceGraph(graph,id,includeImplementation=false) {
  const order=[],cycles=[],seen=new Set(),visiting=new Set();
  const stack=[{id,leave:false}];
  while(stack.length) {
    const current=stack.pop();
    if(current.leave){visiting.delete(current.id);seen.add(current.id);order.push(current.id);continue;}
    if(visiting.has(current.id)){cycles.push(current.id);continue;}
    if(seen.has(current.id))continue;
    visiting.add(current.id);stack.push({id:current.id,leave:true});
    const inputs=inputsFor(graph,current.id,includeImplementation);
    for(let i=inputs.length-1;i>=0;i--)stack.push({id:inputs[i].from,leave:false});
  }
  return {order,cycles:[...new Set(cycles)]};
}

export function nominalBackscatter(data) {
  const value=id=>Number(String(data.nodes.find(n=>n.id===id)?.value||'').split(' ')[0]);
  const fsr=value(cfg('ring.optics.fsr_thz'))*1e12,split=value(cfg('ring.optics.backscatter_splitting_ghz'))*1e9;
  if(!(fsr>0&&split>=0))return null;
  const beta=split/2,delay=1/fsr,rb=2*Math.PI*beta/fsr,g=rb/delay;
  return {fsr,split,beta,delay,rb,g};
}
