import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {defaults,validate,circles,svg,manifest,command} from '../../demo/core.mjs';
for(const c of JSON.parse(readFileSync(new URL('./cases.json',import.meta.url),'utf8'))) test(c.name,()=>{
 const check=()=>validate({...defaults,...c.changes});
 if(c.valid) assert.doesNotThrow(check); else assert.throws(check);
});
test('seed repeats and prompt does not masquerade as inference',()=>{
 assert.deepEqual(circles(defaults),circles({...defaults,prompt:'Something completely different'}));
 assert.notDeepEqual(circles(defaults),circles({...defaults,seed:43}));
 assert.equal(manifest(defaults).backend.model_id,null);
 assert.equal(manifest(defaults).backend.kind,'browser-procedural');
});
test('hostile prompt stays out of SVG and is shell quoted',()=>{
 const s={...defaults,prompt:`it's <script>bad</script> $(echo secret)`};
 assert.ok(!svg(s).includes('<script>'));
 assert.ok(command(s).includes(`'it'\\''s`));
 assert.ok(command(s).includes('--demo'));
});
test('nonfinite values are rejected',()=>{for(const guidance of [NaN,Infinity,-Infinity]) assert.throws(()=>validate({...defaults,guidance}));});
