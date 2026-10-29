import {graphEdges,graphLabel} from './graph-layout.js';

const mathematical=new Set(['equation input','equation constant','recorded input','defined by','function input']);
const contextual=new Set(['model assumption','numerical method','model context']);

export function buildRelationshipIndex(graph) {
  const incoming=new Map(),outgoing=new Map();
  for(const edge of graphEdges(graph)) {
    if(!incoming.has(edge.to))incoming.set(edge.to,[]);
    if(!outgoing.has(edge.from))outgoing.set(edge.from,[]);
    incoming.get(edge.to).push(edge);outgoing.get(edge.from).push(edge);
  }
  return {incoming,outgoing};
}

export function relationshipsFor(graph,index,id) {
  const groups={inputs:[],uses:[],conditions:[],sources:[],codeInputs:[],codeUses:[]};
  for(const edge of index.incoming.get(id)||[]) {
    const key=contextual.has(edge.kind)?'conditions':edge.kind==='source definition'?'sources':
      mathematical.has(edge.kind)&&graph.nodes.get(edge.from).physics?'inputs':'codeInputs';
    groups[key].push({id:edge.from,edge});
  }
  for(const edge of index.outgoing.get(id)||[]) {
    const key=(mathematical.has(edge.kind)||contextual.has(edge.kind))&&graph.nodes.get(edge.to).physics?'uses':'codeUses';
    groups[key].push({id:edge.to,edge});
  }
  for(const key of Object.keys(groups)) {
    const unique=new Map();
    for(const {id:other,edge}of groups[key]) {
      if(!unique.has(other))unique.set(other,{id:other,kinds:[],edges:[]});
      const item=unique.get(other);item.edges.push(edge);if(!item.kinds.includes(edge.kind))item.kinds.push(edge.kind);
    }
    groups[key]=[...unique.values()];
  }
  return groups;
}

export function createRelationshipView(graph,{el,math,unit,onSelect,onInspect},initial) {
  const index=buildRelationshipIndex(graph);
  const root=el('section',{class:'relationship-view','aria-label':'Relationships of the selected quantity'});
  let selected=initial,trail=[initial];
  const pick=id=>onSelect(id);
  const count=items=>items.reduce((sum,item)=>sum+item.edges.length,0);

  function card(item,bucket) {
    const n=graph.nodes.get(item.id),button=el('button',{type:'button',class:'relationship-card','data-relationship-id':n.id,'data-bucket':bucket});
    const top=el('span',{class:'relationship-card-top'});
    if(n.symbol)top.append(math(n.symbol));
    top.append(el('span',{class:'kind-badge'},n.family));button.append(top,el('strong',{},graphLabel(n)));
    if(n.formula_tex)button.append(math(n.formula_tex,true));
    if(n.unit){const row=el('span',{class:'relationship-unit'});row.append(el('span',{},'units'),unit(n.unit));button.append(row);}
    if(n.value&&n.family!=='function')button.append(el('span',{class:'note relationship-value'},'reference value — '+n.value));
    button.append(el('span',{class:'relationship-edge-label'},item.kinds.join(' · ')));
    button.append(el('span',{class:'relationship-follow'},'explore this definition →'));
    button.onclick=()=>pick(n.id);return button;
  }

  function list(items,bucket,empty) {
    const content=el('div',{class:'relationship-list'});
    for(const item of items)content.append(card(item,bucket));
    if(!items.length)content.append(el('p',{class:'relationship-empty note'},empty));
    return content;
  }

  function render() {
    const n=graph.nodes.get(selected),relations=relationshipsFor(graph,index,selected);
    root.replaceChildren();root.dataset.selected=selected;
    root.dataset.relationshipCount=Object.values(relations).flat().reduce((sum,item)=>sum+item.edges.length,0);
    const heading=el('div',{class:'relationship-heading'});
    heading.append(el('h3',{},'one quantity, one step at a time'),el('p',{class:'note'},'read inputs → selected quantity → uses. click any card to follow that relationship. all direct relationships are included; the complete derivation is available below.'));
    const crumbs=el('nav',{class:'relationship-trail','aria-label':'Your exploration path'});
    crumbs.append(el('span',{class:'note'},'your path'));
    for(const [i,id]of trail.entries()) {
      if(i)crumbs.append(el('span',{'aria-hidden':'true'},'→'));
      const node=graph.nodes.get(id),button=el('button',{type:'button',class:'relationship-crumb','aria-current':id===selected?'page':'false',title:graphLabel(node)});
      button.append(node.symbol?math(node.symbol):el('span',{},graphLabel(node)));button.onclick=()=>pick(id);crumbs.append(button);
    }
    root.append(heading,crumbs);

    const lanes=el('div',{class:`relationship-lanes${relations.inputs.length?'':' no-math-inputs'}${relations.uses.length?'':' no-math-uses'}`});
    const inputLane=el('section',{class:'relationship-lane input-lane'});
    inputLane.append(el('h4',{},'what is it built from?'),el('p',{class:'note'},'direct mathematical inputs'),list(relations.inputs,'inputs','no earlier mathematical inputs are recorded here. check the source or model settings below.'));
    const focus=el('section',{class:'relationship-focus'});
    focus.append(el('p',{class:'eyebrow'},'selected quantity'),el('h4',{},graphLabel(n)));
    if(n.symbol&&!n.formula_tex)focus.append(math(n.symbol));
    if(n.formula_tex)focus.append(math(n.formula_tex,true));
    if(n.meaning&&n.meaning!==n.name)focus.append(el('p',{},n.meaning));
    if(n.unit){const row=el('p',{class:'relationship-unit'});row.append(el('span',{},'units'),unit(n.unit));focus.append(row);}
    if(n.value)focus.append(el('p',{class:'note relationship-value'},'indexed reference value — '+n.value));
    if(n.reference_formula){focus.append(el('p',{class:'note'},'with backscatter disabled'),math(n.reference_formula,true));}
    if(n.equivalent||n.scope) {
      const context=el('details',{class:'relationship-focus-context'});context.append(el('summary',{},'model scope & equivalent forms'));
      if(n.scope)context.append(el('p',{class:'note'},n.scope));
      if(n.equivalent){context.append(el('p',{class:'note'},'equivalent form'),math(n.equivalent,true));}
      focus.append(context);
    }
    const inspect=el('button',{type:'button',class:'text-button'},'full definition & source ↓');inspect.onclick=onInspect;focus.append(inspect);
    const useLane=el('section',{class:'relationship-lane use-lane'});
    useLane.append(el('h4',{},'what uses it?'),el('p',{class:'note'},'direct recorded dependents'),list(relations.uses,'uses','no further mathematical dependents are recorded. implementation links below may still show where this definition is used.'));
    lanes.append(inputLane,focus,useLane);root.append(lanes);

    if(relations.conditions.length) {
      const conditions=el('section',{class:'relationship-conditions'});
      conditions.append(el('h4',{},'under which model conditions?'),el('p',{class:'note'},'assumptions and numerical methods provide context; they are not extra operands in the equation.'),list(relations.conditions,'conditions',''));
      root.append(conditions);
    }
    const implementation=el('details',{class:'relationship-implementation'});
    const codeCount=count(relations.sources)+count(relations.codeInputs)+count(relations.codeUses);
    implementation.append(el('summary',{},`where does this connect to the code? · ${codeCount} direct relationships`),el('p',{class:'note'},'source definitions, settings, storage mappings, and static calls are kept separate from mathematical inputs. these links do not prove which branch executed.'));
    for(const [bucket,title,empty]of [['sources','where is it established?','no direct source binding is recorded.'],['codeInputs','which code inputs or settings does it reference?','no other incoming code relationships are recorded.'],['codeUses','which implementation records reference it?','no outgoing code relationships are recorded.']]) {
      const section=el('section',{class:'relationship-code-group'});section.append(el('h4',{},title),list(relations[bucket],bucket,empty));implementation.append(section);
    }
    root.append(implementation);
  }

  render();
  return {element:root,select(id){
    if(id===selected)return;
    selected=id;const previous=trail.indexOf(id);trail=previous>=0?trail.slice(0,previous+1):[...trail,id];render();
  }};
}
