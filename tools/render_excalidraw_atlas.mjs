// export a native atlas through the official Excalidraw renderer.
// dependencies stay outside the simulation environment. see the phase companion README.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

const args=process.argv.slice(2);
function option(name,fallback){const index=args.indexOf(name);return index<0?fallback:args[index+1];}
const packageRoot=option('--packages');
if(!packageRoot)throw Error('supply --packages with a node_modules directory containing Excalidraw and Vite');
const packages=fs.realpathSync(packageRoot);
const require=createRequire(path.join(packages,'package.json'));
const browserRequire=createRequire(path.join(path.resolve(option('--browser-packages',packages)),'package.json'));
const {chromium}=browserRequire('playwright');
const viteRoot=path.dirname(require.resolve('vite/package.json'));
const {createServer}=await import(pathToFileURL(path.join(viteRoot,'dist/node/index.js')));
const root=path.resolve(option('--directory','docs/diagrams/phase'));
const atlasName=option('--atlas','phase-walkthrough.excalidraw');
const manifest=JSON.parse(fs.readFileSync(path.join(root,'manifest.json'),'utf8'));
const temporary=fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(),'excalidraw-export-')));
fs.symlinkSync(packages,path.join(temporary,'node_modules'),'dir');
const fonts=path.join(packages,'@excalidraw/excalidraw/dist/prod/fonts');
fs.mkdirSync(path.join(temporary,'public/assets'),{recursive:true});
fs.cpSync(fonts,path.join(temporary,'public/assets/fonts'),{recursive:true});
fs.writeFileSync(path.join(temporary,'index.html'),'<html><head><meta charset="utf-8"></head><body><script type="module" src="/renderer.js"></script></body></html>');
fs.writeFileSync(path.join(temporary,'renderer.js'),`
window.EXCALIDRAW_ASSET_PATH='/assets/';
const api=await import('@excalidraw/excalidraw');
const font=new FontFace('Excalifont','url(/assets/fonts/Excalifont/Excalifont-Regular-a88b72a24fb54c9f94e3b5fdaa7481c9.woff2)');
document.fonts.add(await font.load());
await document.fonts.ready;
window.renderScene=async(scene,preserve)=>{
 const elements=structuredClone(scene.elements);
 const context=document.createElement('canvas').getContext('2d');
 if(!preserve)for(const element of elements){
  if(element.type!=='text')continue;
  context.font=element.fontSize+'px Excalifont';
  element.width=Math.max(...element.text.split('\\n').map(line=>context.measureText(line).width));
  element.height=element.text.split('\\n').length*element.fontSize*element.lineHeight;
  if(element.customData?.labelCentered)element.x=element.customData.labelAnchorX-element.width/2;
 }
 const svg=await api.exportToSvg({elements,appState:{...scene.appState,exportWithDarkMode:false,exportBackground:true,exportEmbedScene:false},files:scene.files||{},exportPadding:75});
 return {elements,svg:svg.outerHTML,bounds:api.getCommonBounds(elements),fonts:[...document.fonts].filter(f=>f.family==='Excalifont').map(f=>({family:f.family,status:f.status}))};
};
window.ready=true;
`);
const server=await createServer({configFile:false,root:temporary,server:{port:0},logLevel:'error'});
let browser;
try{
 await server.listen();
 const port=server.httpServer.address().port;
 browser=await chromium.launch({headless:true,channel:option('--channel','chrome')});
 const page=await browser.newPage();
 page.on('pageerror',error=>console.error('renderer error',error.message));
 await page.goto(`http://127.0.0.1:${port}`);
 await page.waitForFunction(()=>window.ready,{timeout:60000});
 const preview=await browser.newPage({viewport:{width:2600,height:2000}});
 fs.mkdirSync(path.join(root,'previews'),{recursive:true});
 const rendered=[],report=[];
 for(const item of manifest){
  const file=path.join(root,'drawings',item.slug+'.excalidraw');
  const scene=JSON.parse(fs.readFileSync(file,'utf8'));
  const out=await page.evaluate(async({scene,preserve})=>window.renderScene(scene,preserve),{scene,preserve:!!item.reference});
  if(!item.reference){scene.elements=out.elements;fs.writeFileSync(file,JSON.stringify(scene,null,2)+'\n');}
  fs.writeFileSync(path.join(root,'previews',item.slug+'.svg'),out.svg);
  await preview.setContent('<style>body{margin:0;background:white}svg{display:block;width:2400px;height:auto}</style>'+out.svg);
  await preview.evaluate(()=>document.fonts.ready);
  await preview.locator('svg').screenshot({path:path.join(root,'previews',item.slug+'.png')});
  rendered.push({item,files:scene.files||{},...out});
  report.push({slug:item.slug,bounds:out.bounds,elements:out.elements.length,fonts:out.fonts});
  console.log(item.slug,out.elements.length);
 }
 const cellWidth=Math.max(...rendered.map(d=>d.bounds[2]-d.bounds[0]))+300;
 const cellHeight=Math.max(...rendered.map(d=>d.bounds[3]-d.bounds[1]))+340;
 const all=[],allFiles={};
 rendered.forEach((drawing,index)=>{
  Object.assign(allFiles,drawing.files);
  const [left,top,right,bottom]=drawing.bounds;
  const col=index%3,row=Math.floor(index/3),frameId='atlas-frame-'+index;
  const x=col*cellWidth,y=row*cellHeight;
  all.push({id:frameId,type:'frame',x,y,width:cellWidth-80,height:cellHeight-100,angle:0,strokeColor:'#1e1e1e',backgroundColor:'transparent',fillStyle:'solid',strokeWidth:1,strokeStyle:'solid',roughness:0,opacity:100,groupIds:[],frameId:null,roundness:null,seed:index+1,version:1,versionNonce:index+1,isDeleted:false,boundElements:[],updated:1793664000000,link:null,locked:false,name:drawing.item.slug});
  for(const element of drawing.elements)all.push({...element,x:element.x+x-left+75,y:element.y+y-top+100,frameId});
 });
 fs.writeFileSync(path.join(root,atlasName),JSON.stringify({type:'excalidraw',version:2,source:'mrm-model Excalidraw companion',elements:all,appState:{viewBackgroundColor:'#ffffff',gridSize:null},files:allFiles},null,2)+'\n');
 fs.writeFileSync(path.join(root,'render-report.json'),JSON.stringify(report,null,2)+'\n');
}finally{
 if(browser)await browser.close();
 await server.close();
 fs.rmSync(temporary,{recursive:true,force:true});
}
