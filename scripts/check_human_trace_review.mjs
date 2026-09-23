// Isolated synthetic reviewer interactions; never writes the real review store.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
const require=createRequire(new URL('../frontend/package.json',import.meta.url));
const {JSDOM}=require('jsdom');
const html=readFileSync(new URL('../.local/trace-audit-v1/review/index.html',import.meta.url),'utf8');
let downloaded=null;
function boot(saved){return new JSDOM(html,{runScripts:'dangerously',url:'http://127.0.0.1:8012/',beforeParse(w){
 w.URL.createObjectURL=blob=>{downloaded=blob;return 'blob:review-test';};w.URL.revokeObjectURL=()=>{};
 w.HTMLAnchorElement.prototype.click=()=>{};
 if(saved)for(const [k,v] of Object.entries(saved))w.localStorage.setItem(k,v);
}});}
const dom=boot(),w=dom.window,d=w.document;
const $=s=>d.querySelector(s);
const set=(id,value)=>{d.getElementById(id).value=value;};
const submit=()=>$('#stage-form').dispatchEvent(new w.Event('submit',{cancelable:true,bubbles:true}));
const snapshot=()=>Object.fromEntries(Array.from({length:w.localStorage.length},(_,i)=>{const k=w.localStorage.key(i);return[k,w.localStorage.getItem(k)];}));
const stored=()=>JSON.parse(Object.values(snapshot())[0]);
assert.equal($('#step').textContent,'A · 评价证据');
assert(!$('#workspace').textContent.includes('数据集标注'));
assert(!$('#workspace').textContent.includes('版本 A'));
set('requirements','测试要求');set('sufficiency','insufficient');set('reason','仅为自动化界面检查，不是人工标签');submit();
assert.equal($('#step').textContent,'A · 评价证据');assert($('#message').textContent.includes('审阅者'));
set('reviewer','automated-ui-test-not-a-human');set('exposure','unknown');$('#identity').click();
// Pending notes survive changing pages before stage submission.
set('requirements','页面切换前的草稿');$('#requirements').dispatchEvent(new w.Event('input',{bubbles:true}));
$('#next').click();$('#prev').click();assert.equal($('#requirements').value,'页面切换前的草稿');
set('requirements','测试要求');set('sufficiency','insufficient');set('reason','仅测试阶段锁定');submit();
assert.equal($('#step').textContent,'B · 评价草稿');assert(!$('#workspace').textContent.includes('数据集标注'));
assert.equal($('#requirements'),null);const frozenA=JSON.stringify(stored().records.T01.a);
// T01 contains a generator abstention and one drafted answer.
assert($('#A_abstention'));assert.equal($('#A_supported'),null);
set('A_abstention','yes');set('A_reason','测试：弃答单列');
set('B_supported','yes');set('B_direct','no');set('B_complete','no');set('B_action','full');set('B_reason','测试：目标错位');submit();
assert.equal($('#step').textContent,'B · 评价草稿');assert($('#message').textContent.includes('冲突'));
set('B_action','bounded_partial');submit();assert.equal($('#step').textContent,'B · 评价草稿');assert($('#message').textContent.includes('部分发布'));
set('B_action','reject');submit();assert.equal($('#step').textContent,'C · 解盲对照');
assert($('#workspace').textContent.includes('数据集标注'));assert(stored().records.T01.revealed_at);
set('changed','no');set('A_final','yes');set('B_final','reject');set('reason','自动化测试，无人工裁决');submit();
assert.equal($('#step').textContent,'已完成');assert.equal(JSON.stringify(stored().records.T01.a),frozenA);
const reload=boot(snapshot());assert.equal(reload.window.document.querySelector('#step').textContent,'已完成');reload.window.close();
$('#export').click();assert(downloaded);
const exported=await new Promise((resolve,reject)=>{const reader=new w.FileReader();reader.onload=()=>resolve(JSON.parse(reader.result));reader.onerror=reject;reader.readAsText(downloaded);});
assert.equal(exported.completed_questions,1);assert.equal(exported.cases.filter(c=>c.status==='pending').length,23);
assert.equal(exported.identity_verification,'not_verified');
assert(!JSON.stringify(exported).includes('matched_chunk_ids'));
assert(!JSON.stringify(exported).includes('vanilla transformer'));
assert.equal(exported.cases[0].review.a.values.sufficiency,'insufficient');
// Corrupted storage fails closed rather than silently replacing reviewer records.
const broken=boot({[Object.keys(snapshot())[0]]:'not-json'});
assert(broken.window.document.querySelector('#message').textContent.includes('无法读取'));
assert.equal(broken.window.document.querySelector('#stage-form'),null);broken.window.close();w.close();
console.log('PASS: phase masking, identity requirement, autosave/navigation/reload, immutable initial judgment, separate abstention, conflicting labels, reveal ordering, export exclusions, corrupt-store handling. Synthetic UI actions only.');
