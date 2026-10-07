import {defaults,validate,svg,manifest,command} from './core.mjs';
const $ = id => document.getElementById(id);
const form=$('controls');
let current, imageURL;
function render(settings) {
  current=validate(settings);
  if(imageURL) URL.revokeObjectURL(imageURL);
  imageURL=URL.createObjectURL(new Blob([svg(current)],{type:'image/svg+xml'}));
  $('art').src=imageURL;
  $('art').width=current.width; $('art').height=current.height;
  $('art').alt=`Procedural circles, seed ${current.seed}; no AI inference`;
  $('dimensions').textContent=`${current.width} × ${current.height}`;
  $('seed-label').textContent=`SEED ${current.seed}`;
  $('manifest').textContent=JSON.stringify(manifest(current),null,2);
  $('command').textContent=command(current);
  $('status').textContent='Preview rendered. Downloads match this run.';
}
form.addEventListener('submit',event=>{
  event.preventDefault();
  const data=Object.fromEntries(new FormData(form));
  for(const key of ['seed','width','height','steps','guidance']) data[key]=Number(data[key]);
  try {render(data);} catch(error) {$('status').textContent=error.message;}
});
form.addEventListener('input',()=>{$('status').textContent='Settings changed. Render to update the preview and downloads.';});
$('reset-run').addEventListener('click',()=>{form.reset();render(defaults);});
function download(contents,type,name) {
  const url=URL.createObjectURL(new Blob([contents],{type}));
  const a=document.createElement('a');a.href=url;a.download=name;a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$('save-svg').addEventListener('click',()=>download(svg(current),'image/svg+xml',`prism-demo-${current.seed}.svg`));
$('save-json').addEventListener('click',()=>download(JSON.stringify(manifest(current),null,2),'application/json',`prism-demo-${current.seed}.json`));
render(defaults);
