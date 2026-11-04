const diagrams=window.PHASE_DIAGRAMS;
const topics=document.querySelector('#topics');
const stage=document.querySelector('#stage');
let selected=0;
diagrams.forEach((item,index)=>{
  const link=document.createElement('a');
  link.href=`#${item.slug}`;
  link.textContent=`${String(index).padStart(2,'0')} — ${item.title}`;
  topics.append(link);
});
function show(){
  const requested=decodeURIComponent(location.hash.slice(1));
  selected=Math.max(0,diagrams.findIndex(item=>item.slug===requested));
  const item=diagrams[selected];
  document.title=`${item.title} — phase.py`;
  document.querySelector('#count').textContent=`${selected+1} of ${diagrams.length}`;
  const image=document.querySelector('#preview');
  image.src=`previews/${item.slug}.svg`;
  image.alt=`${item.title}. ${item.subtitle}`;
  document.querySelector('#drawing').href=`drawings/${item.slug}.excalidraw`;
  document.querySelector('#svg').href=`previews/${item.slug}.svg`;
  document.querySelector('#png').href=`previews/${item.slug}.png`;
  document.querySelector('#math').href=`../phase-math/index.html${item.reference?'':'#'+item.slug}`;
  document.querySelector('#note').textContent=item.subtitle;
  document.querySelector('#code').textContent=`code connection — ${item.code.replaceAll('; ',' — ')}`;
  document.querySelector('#previous').disabled=selected===0;
  document.querySelector('#next').disabled=selected===diagrams.length-1;
  [...topics.children].forEach((link,index)=>{
    if(index===selected)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');
  });
  stage.scrollTo(0,0);
}
function move(direction){location.hash=diagrams[Math.max(0,Math.min(diagrams.length-1,selected+direction))].slug;}
document.querySelector('#previous').addEventListener('click',()=>move(-1));
document.querySelector('#next').addEventListener('click',()=>move(1));
document.querySelector('#zoom').addEventListener('click',event=>{
  const enlarged=stage.classList.toggle('enlarged');
  event.currentTarget.setAttribute('aria-pressed',String(enlarged));
  event.currentTarget.textContent=enlarged?'fit diagram':'enlarge';
});
let drag=null;
stage.addEventListener('pointerdown',event=>{
  if(!stage.classList.contains('enlarged')||event.button!==0)return;
  drag={x:event.clientX,y:event.clientY,left:stage.scrollLeft,top:stage.scrollTop};
  stage.setPointerCapture(event.pointerId);stage.classList.add('dragging');
});
stage.addEventListener('pointermove',event=>{
  if(drag){stage.scrollLeft=drag.left+drag.x-event.clientX;stage.scrollTop=drag.top+drag.y-event.clientY;}
});
function endDrag(){drag=null;stage.classList.remove('dragging');}
stage.addEventListener('pointerup',endDrag);stage.addEventListener('pointercancel',endDrag);
window.addEventListener('hashchange',show);
window.addEventListener('keydown',event=>{
  if(event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;
  if(event.key==='ArrowRight'){event.preventDefault();move(1);}
  if(event.key==='ArrowLeft'){event.preventDefault();move(-1);}
});
show();
