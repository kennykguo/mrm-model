import {buildGraph, searchGraph, inputsFor, traceGraph, nominalBackscatter, BACKSCATTER_TARGET} from './derivations.js';
import {createRelationshipView} from './relationships.js';

export async function createExplorer(data, ui) {
  const {el,math,unit,setMode}=ui, $=id=>document.getElementById(id);
  const response=await fetch('source-index.json');
  if(!response.ok)throw Error('source-index.json could not load');
  const graph=buildGraph(data,await response.json());
  let requested='';try{requested=decodeURIComponent(location.hash).replace(/^#derive\//,'');}catch{}
  let selected=graph.nodes.has(requested)?requested:BACKSCATTER_TARGET,layout='graph';
  const previous=[];
  let renderedSearch=null;
  const resultButtons=new Map();
  let universeView,universeLoading=false;
  const nominal=nominalBackscatter(data);
  const values=nominal?new Map([
    ['equation.fsr_si',nominal.fsr],['equation.splitting_si',nominal.split],['equation.beta_nominal',nominal.beta],
    ['equation.round_trip_backscatter',nominal.rb],['equation.round_trip_delay',nominal.delay],[BACKSCATTER_TARGET,nominal.g],
  ]):new Map();
  const science=value=>{
    if(value===0)return '0';
    const [mantissa,exponent]=value.toExponential(3).split('e');
    return `${Number(mantissa)}\\times10^{${Number(exponent)}}`;
  };
  const includeCode=()=>$('trace-code').checked;
  const label=n=>n.family==='function'?n.qualified:n.name;
  const kind=n=>n.family==='physical quantity'?(n.type==='leaf'?'preset anchor':n.type==='derived'?'preset derivation':'stored physical value'):n.family;
  const visualMap=createRelationshipView(graph,{el,math,unit,onSelect:id=>select(id,true,true),onInspect:()=>$('definition-inspector').scrollIntoView({block:'start'})},selected);
  $('visual-map-host').replaceChildren(visualMap.element);

  function select(id,save=true,scroll=false) {
    if(!graph.nodes.has(id))return;
    if(save&&selected!==id)previous.push(selected);
    selected=id;setMode('derive',false);
    $('trace-back').disabled=!previous.length;
    $('derive-selected').textContent=label(graph.nodes.get(id));
    renderFocus(scroll&&layout==='global');renderResults();
    if(save)history.replaceState(null,'',`#derive/${encodeURIComponent(id)}`);
    if(scroll)$(layout==='graph'?'visual-map-host':layout==='global'?'global-map-host':'trace-canvas').scrollIntoView({block:'start'});
  }

  function renderResults(openForSearch=false) {
    const query=$('derive-search').value,category=$('derive-category').value;
    const index=$('definition-index');
    const searching=Boolean(query.trim()||category!=='all');
    if(openForSearch===true&&searching)index.open=true;
    const signature=JSON.stringify([query,category,index.open]);
    if(signature===renderedSearch){for(const [id,button]of resultButtons)button.setAttribute('aria-pressed',id===selected);return;}
    renderedSearch=signature;resultButtons.clear();
    const found=searchGraph(graph,query,category);
    $('derive-count').textContent=searching?`${found.length.toLocaleString()} matches · search the complete index`:`browse all ${graph.nodes.size.toLocaleString()} definitions and records`;
    const result=$('derive-results');result.replaceChildren();
    if(universeView)universeView.highlight(searching?found.map(n=>n.id):[]);
    if(!index.open)return;
    for(const n of found) {
      const button=el('button',{type:'button',class:'definition-result','aria-pressed':n.id===selected});
      const top=el('span',{class:'definition-result-top'});
      if(n.symbol)top.append(math(n.symbol));
      if(n.unit&&n.physics)top.append(unit(n.unit));
      top.append(el('span',{class:'kind-badge'},kind(n)));
      button.append(top,el('strong',{},label(n)));
      if(n.family==='function input'||n.family==='function output')button.append(el('code',{class:'note'},n.qualified));
      if(n.formula_tex)button.append(math(n.formula_tex));
      if(!n.physics)button.append(el('span',{class:'note'},n.source_ref?.path||n.id));
      button.onclick=()=>select(n.id,true,true);resultButtons.set(n.id,button);result.append(button);
    }
    if(!found.length)result.append(el('p',{class:'empty'},'no definitions match. try a symbol, code name, or function name.'));
  }

  function sourceDetails(n, caption='where is this established?') {
    const references=[];
    if(n.source_ref)references.push(n);
    for(const id of n.source_ids||[])if(graph.nodes.has(id))references.push(graph.nodes.get(id));
    if(!references.length)return null;
    const details=el('details',{class:'source-details'});details.append(el('summary',{},caption));
    for(const source of references) {
      const ref=source.source_ref;if(!ref)continue;
      const entry=el('section',{class:'source-entry'});
      entry.append(el('p',{class:'code-source'},`${ref.path} · line ${ref.line}`));
      if(source.signature)entry.append(el('code',{},source.signature));
      if(source.comments?.length) {
        const comments=el('div',{class:'source-comments'});
        comments.append(el('p',{class:'note'},'comments beside this definition'));
        for(const comment of source.comments)comments.append(el('p',{},comment));
        entry.append(comments);
      }
      const pre=el('pre');pre.append(el('code',{},ref.code));entry.append(pre);
      if(source.id!==n.id&&source.family==='function') {
        const open=el('button',{type:'button',class:'text-button'},'trace this source function →');
        open.onclick=()=>select(source.id,true,true);entry.append(open);
      }
      details.append(entry);
    }
    return details;
  }

  function quantityCard(id,{index=null}={}) {
    const n=graph.nodes.get(id),card=el('article',{class:`derivation-card${id===selected?' target-card':''}`,'data-definition':id});
    const top=el('div',{class:'derivation-card-top'}),heading=el('div',{class:'derivation-card-name'});
    if(index!==null)heading.append(el('span',{class:'step-number'},String(index+1).padStart(2,'0')));
    if(n.symbol)heading.append(math(n.symbol));
    heading.append(el('strong',{},label(n)));top.append(heading,el('span',{class:'kind-badge'},kind(n)));
    card.append(top);
    {
      if(n.formula_tex)card.append(math(n.formula_tex,true));
      else if(n.family==='code constant'&&n.value)card.append(el('code',{class:'source-expression'},`${n.name} = ${n.value}`));
      if(n.meaning&&n.meaning!==n.name&&['equation','physical constant','function'].includes(n.family))card.append(el('p',{class:'note'},n.meaning));
      const bottom=el('div',{class:'derivation-card-bottom'});
      if(n.value && n.family!=='code constant')bottom.append(el('span',{class:'note'},`recorded reference value · ${n.value}`));
      if(values.has(id))bottom.append(math(science(values.get(id))));
      if(n.unit)bottom.append(unit(n.unit));
      if(values.has(id))bottom.append(el('span',{class:'note'},'enabled-backscatter example'));
      if(bottom.children.length)card.append(bottom);
      if(id===selected&&n.equivalent)card.append(math(n.equivalent,true));
      if(id===selected){const source=sourceDetails(n);if(source)card.append(source);}
    }
    const controls=el('div',{class:'derivation-card-controls'});
    if(id!==selected){const open=el('button',{type:'button',class:'text-button'},'focus & inspect →');open.onclick=()=>select(id,true,true);controls.append(open);}
    if(n.source_ref){const ref=n.source_ref;controls.append(el('span',{class:'note'},`${ref.path} · ${ref.line}`));}
    if(controls.children.length)card.append(controls);
    return card;
  }

  function readingView(trace) {
    const list=el('div',{class:'derivation-reading'});
    const depth=new Map();
    for(const id of trace.order){const inputs=inputsFor(graph,id,includeCode());depth.set(id,inputs.length?1+Math.max(...inputs.map(e=>depth.get(e.from)||0)):0);}
    const ordered=[...trace.order].sort((a,b)=>depth.get(a)-depth.get(b));
    let layer=-1;
    ordered.forEach((id,index)=>{
      if(depth.get(id)!==layer){layer=depth.get(id);list.append(el('h4',{class:'reading-layer'},layer===0?'starting points · constants, preset inputs, or caller arguments':`layer ${layer} · built from earlier quantities`));}
      const step=el('section',{class:'reading-step'});step.append(quantityCard(id,{index}));
      const inputs=inputsFor(graph,id,includeCode());
      if(inputs.length) {
        const line=el('div',{class:'reading-inputs'});line.append(el('span',{class:'note'},'uses'));
        for(const {from}of inputs){const b=el('button',{type:'button',class:'input-chip'});const n=graph.nodes.get(from);b.append(n.symbol?math(n.symbol):el('span',{},label(n)));b.onclick=()=>select(from,true,true);line.append(b);}
        step.append(line);
      } else step.append(el('p',{class:'trace-boundary'},'start here · '+(graph.nodes.get(id).boundary||'recorded starting point')));
      list.append(step);
    });
    return list;
  }

  function renderInspector() {
    const n=graph.nodes.get(selected),panel=$('definition-inspector');
    panel.replaceChildren(el('p',{class:'eyebrow'},'selected definition'),el('h3',{},label(n)));
    if(n.symbol)panel.append(math(n.symbol));
    if(n.meaning)panel.append(el('p',{},n.meaning));
    if(n.value)panel.append(el('p',{class:'note'},'indexed reference value'),el('pre',{class:'indexed-value'},n.value));
    if(n.formula_tex)panel.append(math(n.formula_tex,true));
    if(n.equivalent){panel.append(el('p',{class:'note'},'equivalent identity — the same quantity, another form'),math(n.equivalent,true));}
    if(n.unit){const u=el('p',{class:'inspector-unit'});u.append(el('span',{class:'note'},'units'),unit(n.unit));panel.append(u);}
    if(n.scope)panel.append(el('p',{class:'definition-scope'},n.scope));
    if(n.reference_formula)panel.append(el('p',{class:'note'},'indexed reference with backscatter disabled'),math(n.reference_formula,true));
    const source=sourceDetails(n);if(source)panel.append(source);
    const mapping=el('details');mapping.append(el('summary',{},'code names & stored representations'),el('code',{},n.id));
    for(const alias of n.aliases||[])mapping.append(el('code',{class:'alias-name'},alias));
    for(const id of n.records||[])if(graph.nodes.has(id)) {
      const b=el('button',{class:'code-link',type:'button'},id);b.onclick=()=>select(id,true,true);mapping.append(b);
    }
    if(n.storage)mapping.append(el('p',{class:'code-source'},n.storage));
    if(n.source && !n.source_ref)mapping.append(el('p',{class:'note'},`recorded provenance · ${n.source}`));
    if(n.note)mapping.append(el('p',{class:'note'},n.note));
    if(n.formula&&!n.formula_tex)mapping.append(el('p',{class:'code-source'},n.formula));
    panel.append(mapping);
    if(n.family==='function') {
      const args=el('details');args.append(el('summary',{},'arguments & returned expressions'),el('code',{},n.signature));
      args.append(el('p',{class:'note'},'argument values come from callers. static call links do not prove which branch ran.'));
      for(const value of n.returns||[])args.append(el('pre',{},value));panel.append(args);
    }
    const uses=(graph.outgoing.get(selected)||[]).filter(e=>includeCode()||n.family==='function'||graph.nodes.get(e.to)?.physics);
    const downstream=el('details');downstream.append(el('summary',{},`what uses this? · ${new Set(uses.map(e=>e.to)).size} recorded dependents`));
    for(const id of new Set(uses.map(e=>e.to))){const b=el('button',{type:'button',class:'code-link'},label(graph.nodes.get(id)));b.onclick=()=>select(id,true,true);downstream.append(b);}panel.append(downstream);
  }

  function renderFocus(focus=false) {
    $('derive-selected').textContent=label(graph.nodes.get(selected));
    const trace=traceGraph(graph,selected,includeCode());
    $('trace-count').textContent=`${trace.order.length} unique definitions in this ancestry${trace.cycles.length?` · ${trace.cycles.length} feedback / recursive references`:''}`;
    $('trace-help').textContent=layout==='global'?'the entire index is available here for reference. use relationships to study one quantity at a time.':'read from top to bottom — starting inputs first, then every quantity built from them. feedback references couple equations; they are not an execution order.';
    $('trace-help').hidden=layout==='graph';$('trace-count').hidden=layout!=='reading';
    $('trace-canvas').replaceChildren(...(layout==='reading'?[readingView(trace)]:[]));
    $('trace-canvas').hidden=layout!=='reading';$('visual-map-host').hidden=layout!=='graph';
    $('global-map-host').hidden=layout!=='global';
    $('trace-tree').setAttribute('aria-pressed',layout==='graph');$('trace-reading').setAttribute('aria-pressed',layout==='reading');$('trace-global').setAttribute('aria-pressed',layout==='global');
    $('trace-code').closest('label').hidden=layout!=='reading';
    visualMap.select(selected);
    if(layout==='global') {
      if(universeView)universeView.select(selected,{focus});
      else if(!universeLoading) {
        universeLoading=true;$('global-map-host').textContent='loading the whole-index overview…';
        import('./graph-view.js').then(({createGraphView})=>{
          universeView=createGraphView(graph,{el,math,unit,onSelect:id=>select(id,true,false),onInspect:()=>$('definition-inspector').scrollIntoView({block:'start'})},selected);
          $('global-map-host').replaceChildren(universeView.element);
          const query=$('derive-search').value,category=$('derive-category').value;
          if(query.trim()||category!=='all')universeView.highlight(searchGraph(graph,query,category).map(n=>n.id));
        }).catch(()=>{$('global-map-host').textContent='the overview could not load. relationships, search, and the complete reading order remain available.';universeLoading=false;});
      }
    }
    renderInspector();
    $('nominal-example').hidden=!values.has(selected);
    if(values.has(selected)) {
      const box=$('nominal-example');box.replaceChildren(el('h4',{},'what would the numbers be with backscatter enabled?'),el('p',{class:'note'},'use the g26 coefficient anchors with backscatter explicitly on and device variation off. this differs from the indexed reference, which disables backscatter. these are illustrative calculations, not the current playback state.'));
      box.append(math(rawExample(nominal),true));
    }
  }

  function rawExample(v) {
    return String.raw`\Delta f_{\mathrm{split}}=${(v.split/1e9).toFixed(2)}\,\mathrm{GHz},\quad\Delta f_{\mathrm{FSR}}=${(v.fsr/1e12).toFixed(2)}\,\mathrm{THz}`+
      String.raw`\qquad r_b=${v.rb.toFixed(5)},\quad t_{\mathrm{rt}}=${(v.delay*1e12).toFixed(3)}\,\mathrm{ps},\quad g=${science(v.g)}\,\mathrm{rad/s}`;
  }

  $('derive-search').oninput=()=>renderResults(true);
  $('definition-index').ontoggle=renderResults;
  $('derive-category').onchange=()=>renderResults(true);
  $('trace-code').onchange=()=>renderFocus();
  $('trace-tree').onclick=()=>{layout='graph';renderFocus();$('visual-map-host').scrollIntoView({block:'start'});};
  $('trace-reading').onclick=()=>{layout='reading';renderFocus();};
  $('trace-global').onclick=()=>{layout='global';renderFocus();};
  $('derive-example').onclick=()=>{$('derive-search').value='g';$('derive-category').value='all';select(BACKSCATTER_TARGET,true,true);};
  $('derive-example').replaceChildren(el('span',{},'try the '),math(String.raw`g=\frac{r_b}{t_{\mathrm{rt}}}`),el('span',{},' example'));
  $('trace-back').onclick=()=>{const id=previous.pop();if(id){select(id,false,true);history.replaceState(null,'',`#derive/${encodeURIComponent(id)}`);}};
  $('derive-search').onkeydown=event=>{if(event.key==='Enter'){const first=searchGraph(graph,$('derive-search').value,$('derive-category').value)[0];if(first)select(first.id,true,true);}};
  $('derive-summary').textContent=`${data.nodes.length} model records · ${graph.meta.functions.toLocaleString()} source functions · ${graph.meta.constants.toLocaleString()} source constants · ${graph.meta.contracts} documented function quantities · ${graph.meta.files} Python files`;
  renderResults();renderFocus();
  return {select,has:id=>graph.nodes.has(id),selected:()=>selected};
}
