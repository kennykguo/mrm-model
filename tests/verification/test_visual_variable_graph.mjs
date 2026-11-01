import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import {buildGraph,traceGraph,BACKSCATTER_TARGET} from '../../docs/variables/derivations.js';
import {graphEdges,layoutUniverse,layoutDerivation,routeEdge} from '../../docs/variables/graph-layout.js';
import {MODEL_CONTEXTS} from '../../docs/variables/model-context.js';
import {buildRelationshipIndex,relationshipsFor} from '../../docs/variables/relationships.js';

const root=fileURLToPath(new URL('../../',import.meta.url));
const graph=buildGraph(JSON.parse(readFileSync(root+'docs/variables/data.json')),JSON.parse(readFileSync(root+'docs/variables/source-index.json')));
const layout=layoutUniverse(graph),edges=graphEdges(graph);

function noOverlap(positions) {
  const sorted=[...positions.values()].sort((a,b)=>a.x-b.x),active=[];
  for(const p of sorted){
    for(let i=active.length-1;i>=0;i--)if(active[i].x+active[i].width<=p.x)active.splice(i,1);
    for(const other of active)assert.ok(p.y+p.height<=other.y||other.y+other.height<=p.y,`${p.id} overlaps ${other.id}`);
    active.push(p);
  }
}

test('the universe layout retains every node and every relationship',()=>{
  assert.equal(layout.positions.size,graph.nodes.size);
  for(const id of graph.nodes.keys())assert.ok(layout.positions.has(id),id);
  const expected=new Set();
  for(const entries of graph.incoming.values())for(const e of entries)expected.add(JSON.stringify([e.from,e.to,e.kind]));
  for(const e of edges)expected.delete(JSON.stringify([e.from,e.to,e.kind]));
  assert.equal(expected.size,0);
  for(const [index,e]of edges.entries())assert.ok(routeEdge(e,layout,index),e.from+' -> '+e.to);
  noOverlap(layout.positions);
});

test('orthogonal connectors do not enter any node rectangle',()=>{
  const cell=1000,bins=new Map();
  for(const p of layout.positions.values())for(let x=Math.floor(p.x/cell);x<=Math.floor((p.x+p.width)/cell);x++)for(let y=Math.floor(p.y/cell);y<=Math.floor((p.y+p.height)/cell);y++){
    const key=x+','+y;if(!bins.has(key))bins.set(key,[]);bins.get(key).push(p);
  }
  edges.forEach((edge,index)=>{
    const points=routeEdge(edge,layout,index).points;
    for(let i=1;i<points.length;i++){
      const a=points[i-1],b=points[i];assert.ok(a.x===b.x||a.y===b.y);
      const candidates=new Set();
      for(let x=Math.floor(Math.min(a.x,b.x)/cell);x<=Math.floor(Math.max(a.x,b.x)/cell);x++)for(let y=Math.floor(Math.min(a.y,b.y)/cell);y<=Math.floor(Math.max(a.y,b.y)/cell);y++)for(const p of bins.get(x+','+y)||[])candidates.add(p);
      for(const p of candidates){
        const intersects=a.x===b.x?a.x>p.x&&a.x<p.x+p.width&&Math.max(a.y,b.y)>p.y&&Math.min(a.y,b.y)<p.y+p.height:
          a.y>p.y&&a.y<p.y+p.height&&Math.max(a.x,b.x)>p.x&&Math.min(a.x,b.x)<p.x+p.width;
        assert.ok(!intersects,`${edge.from} -> ${edge.to} crosses ${p.id}`);
      }
    }
  });
});

test('a selected derivation retains all ancestors and marks model conditions',()=>{
  const trace=traceGraph(graph,BACKSCATTER_TARGET,true),ids=new Set(trace.order);
  const selectedEdges=edges.filter(e=>ids.has(e.from)&&ids.has(e.to)&&e.kind!=='source definition');
  const selected=layoutDerivation(graph,trace.order,selectedEdges);
  assert.equal(selected.positions.size,trace.order.length);noOverlap(selected.positions);
  assert.ok(ids.has('model.tcmt'));assert.ok(ids.has('cfg.fidelity.backscatter'));
  assert.match(graph.nodes.get(BACKSCATTER_TARGET).reference_formula,/=0/);
  assert.ok(!edges.some(e=>e.from==='equation.splitting_si'&&e.to==='pp.optics.backscatter_splitting_hz'&&e.kind==='defined by'));
  for(const context of MODEL_CONTEXTS)for(const id of [...context.inputs,...context.source_ids,...context.applies_to])assert.ok(graph.nodes.has(id),context.id+' -> '+id);
});

test('traversal has no fixed depth or neighbour cutoff',()=>{
  const size=6000,nodes=new Map(),incoming=new Map();
  for(let i=0;i<size;i++){const id=String(i);nodes.set(id,{id,physics:true,family:'physical quantity'});incoming.set(id,i?[{from:String(i-1),to:id,kind:'recorded input'}]:[]);}
  const trace=traceGraph({nodes,incoming},String(size-1),true);
  assert.equal(trace.order.length,size);assert.equal(trace.order[0],'0');
});

test('viewer code has no result pagination, neighbour cap, or collapsed ancestry',()=>{
  const code=['map.js','explorer.js'].map(file=>readFileSync(root+'docs/variables/'+file,'utf8')).join('\n');
  assert.doesNotMatch(code,/symbolLimit|modelLimit|found\.slice\(|sorted\.slice\(|derive-more|more-symbols|more-model|path\.size\s*</);
});

test('the focused view retains every direct relationship across the full index',()=>{
  const index=buildRelationshipIndex(graph);
  for(const id of graph.nodes.keys()) {
    const actual=Object.values(relationshipsFor(graph,index,id)).flat().flatMap(item=>item.edges);
    const expected=[...index.incoming.get(id)||[],...index.outgoing.get(id)||[]];
    assert.equal(actual.length,expected.length,id);
    const key=e=>JSON.stringify([e.from,e.to,e.kind]);
    assert.deepEqual(actual.map(key).sort(),expected.map(key).sort(),id);
  }
});

test('the focused backscatter view separates operands, assumptions, and source bindings',()=>{
  const index=buildRelationshipIndex(graph),g=relationshipsFor(graph,index,BACKSCATTER_TARGET);
  assert.deepEqual(g.inputs.map(item=>item.id).sort(),['equation.round_trip_backscatter','equation.round_trip_delay'].sort());
  assert.deepEqual(g.conditions.map(item=>item.id),['model.tcmt']);
  assert.ok(g.sources.some(item=>item.id==='fn:mrm.plant.optics.tcmt_backscatter_rate'));
  assert.ok(!g.inputs.some(item=>item.id.startsWith('fn:')||item.id.startsWith('model.')));
  const rb=relationshipsFor(graph,index,'equation.round_trip_backscatter');
  assert.ok(rb.uses.some(item=>item.id===BACKSCATTER_TARGET));
  const beta=relationshipsFor(graph,index,'equation.beta_nominal');
  assert.ok(beta.codeInputs.some(item=>item.id==='cfg.fidelity.backscatter'));
});
