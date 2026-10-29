import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import {buildGraph,searchGraph,traceGraph,inputsFor,nominalBackscatter,DEFINITIONS,BACKSCATTER_TARGET} from '../../docs/variables/derivations.js';
import {normalizeTex} from '../../docs/variables/physics.js';

const root=fileURLToPath(new URL('../../',import.meta.url));
const data=JSON.parse(readFileSync(root+'docs/variables/data.json'));
const sources=JSON.parse(readFileSync(root+'docs/variables/source-index.json'));
const graph=buildGraph(data,sources);

test('all original records and source definitions remain searchable',()=>{
  for(const n of [...data.nodes,...sources.nodes])assert.ok(graph.nodes.has(n.id),n.id);
  for(const q of ['g','g=\\frac{r_b}{t_\\mathrm{rt}}','tcmt_backscatter_rate'])assert.equal(searchGraph(graph,q)[0].id,BACKSCATTER_TARGET);
  assert.equal(searchGraph(graph,'r_b')[0].id,'equation.round_trip_backscatter');
  assert.equal(searchGraph(graph,'t_rt')[0].id,'equation.round_trip_delay');
  assert.equal(searchGraph(graph,'C0')[0].id,'constant.c');
  assert.ok(searchGraph(graph,'dark current','physics').length>0);
  assert.ok(searchGraph(graph,'P.th_ring_obs','implementation').some(n=>n.id==='P.th_ring_obs'));
});

test('backscatter traces to preset anchors, geometry, and physical constants',()=>{
  const trace=traceGraph(graph,BACKSCATTER_TARGET),index=id=>trace.order.indexOf(id);
  assert.deepEqual(trace.cycles,[]);
  for(const id of ['cfg.ring.optics.fsr_thz','cfg.ring.optics.backscatter_splitting_ghz','cfg.ring.optics.n_g','cfg.derived.optics.length_m','constant.c','constant.pi'])assert.ok(index(id)>=0,id);
  assert.ok(index('equation.beta_nominal')<index('equation.round_trip_backscatter'));
  assert.ok(index('equation.round_trip_delay')<index(BACKSCATTER_TARGET));
  assert.equal(new Set(trace.order).size,trace.order.length);
  assert.ok(traceGraph(graph,'pp.optics.backscatter_rb',true).order.includes('equation.round_trip_backscatter'));
});

test('nominal values close the per-trip and per-second identities',()=>{
  const {fsr,split,beta,delay,rb,g}=nominalBackscatter(data);
  assert.equal(beta,split/2);
  assert.ok(Math.abs(fsr*delay-1)<1e-14);
  assert.ok(Math.abs(g/(2*Math.PI*beta)-1)<1e-14);
  assert.ok(Math.abs(rb-g*delay)<1e-14);
  assert.ok(Math.abs(delay-5.555555555555556e-13)<1e-25);
});

test('reviewed equations point to real source locations',()=>{
  for(const d of DEFINITIONS){
    for(const id of [...d.inputs,...d.source_ids||[]])assert.ok(graph.nodes.has(id),`${d.id} -> ${id}`);
    for(const id of d.source_ids||[]){const n=graph.nodes.get(id),lines=readFileSync(root+n.source_ref.path,'utf8').split('\n');assert.ok(lines[n.source_ref.line-1].includes(n.name),id);}
  }
  const g=graph.nodes.get('fn:mrm.plant.optics.tcmt_backscatter_rate');
  assert.match(g.source_ref.code,/return rb \/ t_rt_s/);
  const delay=graph.nodes.get('cfg.derived.optics.round_trip_time_s');
  assert.ok(delay.source_ids.includes('fn:mrm.params.relations._solve_r3_geometry'));
});

test('a source boundary and a feedback cycle are explicit and finite',()=>{
  const n=id=>({id,family:'physical quantity',physics:true});
  const example={nodes:new Map(['a','b','input'].map(id=>[id,n(id)])),incoming:new Map([
    ['a',[{from:'b',to:'a',kind:'recorded input'}]],['b',[{from:'a',to:'b',kind:'recorded input'},{from:'input',to:'b',kind:'recorded input'}]],['input',[]],
  ])};
  assert.deepEqual(traceGraph(example,'a').cycles,['a']);
  assert.equal(traceGraph(example,'a').order.length,3);
  assert.equal(inputsFor(example,'input').length,0);
});

test('all authored, recorded, and extracted mathematical notation renders',()=>{
  const require=createRequire(import.meta.url),katex=require(root+'tools/katex/node_modules/katex');
  for(const n of graph.nodes.values())for(const key of ['symbol','formula_tex','equivalent','reference_formula'])if(n[key]){
    const tex=normalizeTex(n[key]).replace(/\\mathrm\b/g,String.raw`\mathtt`);
    assert.doesNotThrow(()=>katex.renderToString(String.raw`\mathtt{`+tex+'}',{throwOnError:true,trust:false}),`${n.id} ${key}`);
  }
  const contract=graph.nodes.get('contract:mrm.plant.optics.tcmt_backscatter_rate:return');
  assert.ok(contract.physics);
  assert.match(contract.formula_tex,/\\frac/);
  assert.equal(inputsFor(graph,contract.id).length,2);
});
