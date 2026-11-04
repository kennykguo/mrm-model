const sheets=window.PHASE_DIAGRAMS;
const nav=document.querySelector('#topics');
const stage=document.querySelector('#stage');
let selected=0;
for(const item of sheets){const link=document.createElement('a');link.href=`#${item.slug}`;link.textContent=`${item.slug.slice(0,2)} — ${item.title}`;nav.append(link);}
function show(){
 selected=Math.max(0,sheets.findIndex(item=>item.slug===decodeURIComponent(location.hash.slice(1))));
 const item=sheets[selected];
 document.title=`${item.title} — phase.py equations`;
 document.querySelector('#count').textContent=`${selected+1} of ${sheets.length}`;
 const image=document.querySelector('#preview');image.src=`previews/${item.slug}.svg`;image.alt=item.title;
 document.querySelector('#drawing').href=`drawings/${item.slug}.excalidraw`;
 document.querySelector('#latex').href=`latex/${item.slug}.tex`;
 document.querySelector('#svg').href=`previews/${item.slug}.svg`;
 document.querySelector('#concept').href=`../phase/index.html#${item.slug}`;
 document.querySelector('#previous').disabled=selected===0;document.querySelector('#next').disabled=selected===sheets.length-1;
 [...nav.children].forEach((link,index)=>{if(index===selected)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');});
 stage.scrollTo(0,0);
}
function move(delta){location.hash=sheets[Math.max(0,Math.min(sheets.length-1,selected+delta))].slug;}
document.querySelector('#previous').addEventListener('click',()=>move(-1));document.querySelector('#next').addEventListener('click',()=>move(1));
document.querySelector('#zoom').addEventListener('click',event=>{const large=stage.classList.toggle('enlarged');event.currentTarget.setAttribute('aria-pressed',String(large));event.currentTarget.textContent=large?'fit':'enlarge';});
let drag=null;
stage.addEventListener('pointerdown',event=>{if(!stage.classList.contains('enlarged')||event.button!==0)return;drag={x:event.clientX,left:stage.scrollLeft};stage.setPointerCapture(event.pointerId);stage.classList.add('dragging');});
stage.addEventListener('pointermove',event=>{if(drag)stage.scrollLeft=drag.left+drag.x-event.clientX;});
for(const name of ['pointerup','pointercancel'])stage.addEventListener(name,()=>{drag=null;stage.classList.remove('dragging');});
window.addEventListener('hashchange',show);
window.addEventListener('keydown',event=>{if(event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;if(event.key==='ArrowRight'){event.preventDefault();move(1);}if(event.key==='ArrowLeft'){event.preventDefault();move(-1);}});
show();
