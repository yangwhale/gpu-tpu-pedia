/* tpuguru 工作台。对应 README §4（对话 + 配置 + 命令三者联动）与 §6（报告）。 */
'use strict';

const $ = s => document.querySelector(s);
const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c;
  if (h !== undefined) e.innerHTML = h; return e; };
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const md = s => esc(s)
  .replace(/`([^`\n]+)`/g, '<code>$1</code>')
  .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  .split(/\n{2,}/).map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
const fmt = (n, d = 2) => (n === null || n === undefined) ? '—' : Number(n).toFixed(d);

let S = null;              // 当前 state
let TOPOS = {};
let FAMILIES = [];
let BACKENDS = [];
let busy = false;
let hideVoided = false;
let everSawReport = false;
let CLUSTER = null;

/* ── 参数控件表（PARAMS.md 的子集，问号文案三段式）───────── */
const FIELDS = [
  { k: '__topology', label: '目标拓扑', type: 'topo', req: true,
    q: '决定 AOT 按多少张卡编译。名字里的数字是 device，v7 是 2 device/chip —— tpu7x-128 是 64 芯片，不是 128。选错了整份显存结论都不成立。' },
  { k: 'per_device_batch_size', label: 'per_device_batch_size', type: 'num',
    q: '每个 device（不是每芯片）的 batch。它是显存的主要旋钮，但显存不随它单调 —— 实测有 13 超 0.77G、降到 12 反而超 1.26G 的情况，逐档试不要外推。' },
  { k: 'ici_fsdp_parallelism', label: 'ici_fsdp_parallelism', type: 'num',
    q: 'FSDP 宽度，-1 表示吃满剩余 device。每卡常驻权重 ∝ 1/FSDP，是最有效的显存杠杆。装不下先加宽它，再考虑重算。' },
  { k: 'ici_expert_parallelism', label: 'ici_expert_parallelism', type: 'num',
    q: '专家并行 EP。实测大幅负收益：64 芯片 EP=2 掉 39.6%，16 芯片 EP=4 掉 71%。除了 all-to-all 多跳，它还逼着 FSDP 减半。默认填 1。' },
  { k: 'max_target_length', label: 'max_target_length', type: 'num',
    q: '序列长度。影响激活显存与 attention 占比。改了它 tile 甜点值也要重找。' },
  { k: 'quantization', label: 'quantization', type: 'sel', opts: ['', 'fp8', 'fp8_full', 'int8'],
    q: '计算精度。FP8 相对 BF16 大约只快 9% —— fp32 主权重那块不随计算精度缩小，FP8 只压了激活和矩阵乘输入。' },
  { k: 'weight_quantization_calibration_method', label: '量化校准', type: 'sel',
    opts: ['', 'absmax', 'fixed,-224,224'],
    q: 'fixed 是静态 scale，会损害收敛质量，唯一正当理由是配合跨卡量化收集（各卡 scale 必须一致）。不开 QAG 就用 absmax。代价是吃掉两档 batch。' },
  { k: 'weight_dtype', label: 'weight_dtype', type: 'sel', opts: ['', 'float32', 'bfloat16'],
    q: '★ 主权重（优化器里的真身）的存储精度，不是计算精度。bf16 只有 8 位尾数，权重更新会被直接舍掉 —— 不报错但训练是废的。保持 float32。' },
  { k: 'use_tokamax_gmm', label: 'use_tokamax_gmm', type: 'bool',
    q: '切到 tokamax kernel。注意它在两个精度下走的是完全不同的代码：BF16 是裸 ragged_dot（慢 12 倍），FP8 才是 megablox+tokamax 后端。' },
  { k: 'shard_exp_on_fsdp', label: 'shard_exp_on_fsdp', type: 'bool',
    q: '★ 按专家维切权重。开了它而不开 tokamax 会静默漏算 —— kernel 只对本地专家建组元数据，其余完全不参与计算，不报错、loss 照降。默认别开。' },
  { k: 'megablox', label: 'megablox', type: 'bool', q: '用 megablox 的分组矩阵乘 kernel（dropless MoE）。MoE 模型基本都要开。' },
  { k: 'sparse_matmul', label: 'sparse_matmul', type: 'bool', q: 'MoE 走稀疏矩阵乘路径，跟 megablox 配套。' },
  { k: 'num_decoder_layers', label: '层数', type: 'num',
    q: '★ 调配置可以用 4–8 层跑通（快 3 倍，配置错误照样暴露），但**问显存必须换回生产层数** —— 常驻权重是所有层的和。' },
];

/* ── API ─────────────────────────────────────────────────── */
async function api(path, body) {
  const opt = body ? { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body) } : {};
  const r = await fetch(path.replace(/^\//, ''), opt);
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || d.error || r.statusText);
  return d;
}

function toast(t) {
  const n = $('#toast'); n.textContent = t; n.classList.add('on');
  clearTimeout(n._t); n._t = setTimeout(() => n.classList.remove('on'), 2400);
}

function setBusy(b, label) {
  busy = b;
  $('#btn-send').disabled = b; $('#btn-run').disabled = b;
  $('#btn-run').textContent = b && label ? label : '跑 AOT';
}

/* ── 渲染：对话 ───────────────────────────────────────────── */
function renderTurns() {
  const box = $('#turns'); box.innerHTML = '';
  if (!S.turns.length) {
    box.appendChild(el('div', 'empty',
      '<span class="ic">💬</span>把训练命令整段贴进来<br>它会认参数、查已知的坑、生成 AOT 命令'));
  }
  S.turns.forEach((t, i) => {
    const d = el('div', 'turn ' + t.role);
    if (t.role !== 'system') d.appendChild(el('div', 'who', t.role === 'user' ? '你' : 'tpuguru'));
    else d.appendChild(el('div', 'who', '·'));
    // 长命令别 wrap 成散文 —— 等宽 + 保留换行 + 可滚动，能一眼看出它是条命令
    const isCmd = t.role === 'user' && (t.text.includes('\n') || t.text.length > 160)
      && (t.text.includes('python') || t.text.split('=').length > 4);
    let b;
    if (isCmd) {
      b = el('div', 'body');
      const lines = t.text.split('\n').length;
      b.appendChild(el('div', 'cmdmeta', `训练命令 · ${lines} 行 / ${t.text.length} 字符`));
      b.appendChild(el('div', 'cmdblob', esc(t.text)));
    } else {
      b = el('div', 'body', md(t.text));
    }
    if (t.proposal) b.appendChild(renderDiff(t.proposal, i));
    d.appendChild(b); box.appendChild(d);
  });
  box.scrollTop = box.scrollHeight;
}

function renderDiff(p, idx) {
  const d = el('div', 'diff' + (p.applied ? ' applied' : ''));
  p.diff.forEach(x => {
    const it = el('div', 'item');
    it.appendChild(el('div', 'k', esc(x.param)));
    it.appendChild(el('div', 'chg',
      `<span class="from">${esc(x.from ?? '(未设)')}</span> → <span class="to">${esc(x.to)}</span>`));
    if (x.reason) it.appendChild(el('div', 'why', esc(x.reason)));
    d.appendChild(it);
  });
  if (p.applied) {
    d.appendChild(el('div', 'applied-tag', '✓ 已应用'));
  } else {
    const acts = el('div', 'acts');
    const a = el('button', 'btn-primary btn-sm', '应用');
    a.onclick = async () => { setBusy(true); try { S = await api('/api/apply',
      { session_id: S.session_id, diff: p.diff }); render(); } finally { setBusy(false); } };
    const b = el('button', 'btn-ghost btn-sm', '忽略');
    b.onclick = () => { p.applied = null; renderTurns(); };
    acts.append(a, b); d.appendChild(acts);
  }
  return d;
}

/* ── 渲染：配置 ───────────────────────────────────────────── */
function renderCfg() {
  const p = $('#pane-cfg'); p.innerHTML = '';
  p.appendChild(cardPick());

  // lint 卡
  const lc = el('div', 'card');
  lc.appendChild(el('h3', null, '配置体检'));
  const empty = !Object.keys(S.params).length;
  if (empty) {
    lc.appendChild(el('div', 'hint',
      '还没有配置 —— 左边贴一段训练命令，或者直接在下面填。<br>'
      + '<span style="color:var(--faint)">这里现在是空的，不代表「没问题」。</span>'));
  } else if (!S.lint.length) {
    lc.appendChild(el('div', 'hint', '✅ 没有命中已知陷阱。规则库有 9 条，随着踩坑继续长。'));
  } else {
    lc.appendChild(el('div', 'hint', `命中 ${S.lint.length} 条。规则来自实测踩过的坑，不是通用建议。`));
    S.lint.forEach(f => {
      const ic = { fatal: '🔴', warn: '🟡', info: '🔵' }[f.severity] || '•';
      const d = el('div', 'lint ' + f.severity + (f.severity === 'info' ? ' fold' : ''));
      if (f.severity === 'info') d.onclick = () => d.classList.toggle('fold');
      d.appendChild(el('div', 'ic', ic));
      const t = el('div');
      t.appendChild(el('div', 't', esc(f.title) + ` <span class="rule">${esc(f.rule)}</span>`));
      t.appendChild(el('div', 'd', esc(f.detail)));
      if (f.fix) t.appendChild(el('div', 'fix', '→ ' + esc(f.fix)));
      if (f.evidence) t.appendChild(el('div', 'ev', esc(f.evidence)));
      d.appendChild(t); lc.appendChild(d);
    });
  }
  p.appendChild(lc);

  // 参数表单
  const fc = el('div', 'card');
  fc.appendChild(el('h3', null, '关键参数'));
  fc.appendChild(el('div', 'hint',
    `共 ${Object.keys(S.params).length} 个参数、${Object.keys(S.xla_flags).length} 个 XLA flag。`
    + `下面只列常调的，鼠标移到 ⓘ 看它是干什么的。FSDP 实际宽度 <b>${S.fsdp_width || '—'}</b>。`));
  const g = el('div', 'grid');
  FIELDS.forEach(f => g.appendChild(renderField(f)));
  fc.appendChild(g);
  p.appendChild(fc);

  // XLA flags
  if (Object.keys(S.xla_flags).length) {
    const xc = el('div', 'card');
    xc.appendChild(el('h3', null, 'XLA flags'));
    xc.appendChild(el('div', 'hint', '从 LIBTPU_INIT_ARGS / XLA_FLAGS 里拆出来的，转成 compile_xla_flags 传给 AOT。'));
    const rows = Object.entries(S.xla_flags).map(([k, v]) =>
      `<tr><td class="mono">${esc(k)}</td><td class="mono">${esc(v === true ? '(on)' : v)}</td></tr>`).join('');
    xc.appendChild(el('table', 't', `<thead><tr><th>flag</th><th>值</th></tr></thead><tbody>${rows}</tbody>`));
    p.appendChild(xc);
  }

  // 其它参数
  const known = new Set(FIELDS.map(f => f.k));
  const rest = Object.entries(S.params).filter(([k]) => !known.has(k));
  if (rest.length) {
    const rc = el('div', 'card');
    rc.appendChild(el('h3', null, `其余参数 · ${rest.length}`));
    rc.appendChild(el('div', 'hint', '原样透传给 AOT，不做解释。'));
    rc.appendChild(el('table', 't', '<tbody>' + rest.map(([k, v]) =>
      `<tr><td class="mono">${esc(k)}</td><td class="mono">${esc(v)}</td></tr>`).join('') + '</tbody>'));
    p.appendChild(rc);
  }
}

/* 模型族 → 型号 → MoE 后端。**先选它们，后面的映射才成立**：
   tile 上界看 hidden/mlp、整除类 lint 看专家数、显存估算看参数量。 */
function cardPick() {
  const c = el('div', 'card');
  c.appendChild(el('h3', null, '模型与后端'));
  c.appendChild(el('div', 'hint',
    '先选这两项 —— 形状不知道，tile 上界、整除检查、显存估算全都退化成猜。'));

  const cur = S.params.model_name || '';
  const curModel = FAMILIES.flatMap(f => f.models).find(m => m.id === cur);
  const curFam = curModel ? FAMILIES.find(f => f.models.some(m => m.id === cur)).id : '';

  const g = el('div', 'pick');

  // 族
  const fd = el('div', 'field');
  const fl = el('label', null, '模型族');
  const fq = el('span', 'q', 'ⓘ');
  fq.title = 'MaxText 支持的模型族。Hunyuan3 是我们后加进去的，主线没有；其余是主线自带。';
  fl.appendChild(fq); fd.appendChild(fl);
  const fs = el('select');
  fs.appendChild(el('option', null, '— 请选择 —')).value = '';
  FAMILIES.forEach(f => { const o = el('option', null, f.label); o.value = f.id;
    if (f.id === curFam) o.selected = true; fs.appendChild(o); });
  fs.onchange = async () => {
    const fam = FAMILIES.find(x => x.id === fs.value);
    if (fam && fam.models.length) await setParam('model_name', fam.models[0].id);
    else render();
  };
  fd.appendChild(fs); g.appendChild(fd);

  // 型号
  const md = el('div', 'field' + (cur ? '' : ' miss'));
  const ml = el('label', null, '型号');
  const mq = el('span', 'q', 'ⓘ');
  mq.title = '决定层数 / 专家数 / hidden / mlp / 参数量。选错了整份显存与 lint 结论都不成立。';
  ml.appendChild(mq); md.appendChild(ml);
  const ms = el('select');
  const fam = FAMILIES.find(x => x.id === (fs.value || curFam));
  ms.appendChild(el('option', null, fam ? '— 请选择 —' : '— 先选模型族 —')).value = '';
  (fam ? fam.models : []).forEach(m => { const o = el('option', null, m.label); o.value = m.id;
    if (m.id === cur) o.selected = true; ms.appendChild(o); });
  ms.disabled = !fam;
  ms.onchange = () => setParam('model_name', ms.value);
  md.appendChild(ms); g.appendChild(md);

  // MoE 后端
  const isMoe = curModel ? curModel.moe : true;
  const bd = el('div', 'field');
  const bl = el('label', null, 'MoE 后端');
  const bq = el('span', 'q', 'ⓘ');
  bq.title = '选的是「走哪条 kernel 路径」，不是记 flag 名字。选完会一次性把对应参数写进配置。';
  bl.appendChild(bq); bd.appendChild(bl);
  const bs = el('select');
  bs.appendChild(el('option', null, isMoe ? '— 未指定 —' : '— dense 模型不适用 —')).value = '';
  BACKENDS.forEach(b => { const o = el('option', null, b.label); o.value = b.id;
    if (b.id === S.backend) o.selected = true; bs.appendChild(o); });
  bs.disabled = !isMoe;
  bs.onchange = () => setParam('__backend', bs.value);
  bd.appendChild(bs); g.appendChild(bd);
  c.appendChild(g);

  if (curModel) {
    const i = el('div', 'modelinfo');
    const r1 = el('div', 'row1');
    r1.appendChild(el('b', null, esc(curModel.label)));
    r1.appendChild(el('span', 'prov ' + curModel.provenance,
      curModel.provenance === 'measured' ? '我们在 v7 上实测过' : '公开 config · 无我方实测'));
    i.appendChild(r1);
    i.appendChild(el('div', 'shapes',
      `${curModel.layers} 层 · ` + (curModel.moe
        ? `${curModel.num_experts} 专家 · ` : 'dense · ')
      + `hidden ${curModel.hidden} · mlp ${curModel.mlp} · ${curModel.params_b}B 参数`));
    if (curModel.tile_default) i.appendChild(el('div', 'shapes',
      `tile 安全起点 (${curModel.tile_default.join(', ')}) —— 不能超过对应维度`));
    if (curModel.note) i.appendChild(el('div', null, esc(curModel.note)));
    if (curModel.provenance === 'public') i.appendChild(el('div', 'warn',
      '⚠️ 这个模型我们没有 v7 实测数据。别把别的模型的 batch 上限、tile 值直接搬过来。'));
    c.appendChild(i);
  }

  const be = BACKENDS.find(b => b.id === S.backend);
  if (be && isMoe) {
    const i = el('div', 'modelinfo');
    i.appendChild(el('div', null, md2(be.desc)));
    const pr = el('div', 'bepros');
    const mk = (cls, h, arr) => { const d = el('div', cls);
      d.appendChild(el('div', 'h', h));
      d.appendChild(el('ul', null, arr.map(x => `<li>${md2(x)}</li>`).join(''))); return d; };
    pr.appendChild(mk('good', '拿到什么', be.pros));
    pr.appendChild(mk('bad', '代价 / 风险', be.cons));
    i.appendChild(pr);
    c.appendChild(i);
  }
  return c;
}

const md2 = s => esc(s).replace(/`([^`]+)`/g, '<code>$1</code>')
  .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

async function setParam(k, v) {
  setBusy(true);
  try { S = await api('/api/set', { session_id: S.session_id, param: k, value: v }); render(); }
  catch (e) { toast('改失败：' + e.message); } finally { setBusy(false); }
}

function renderField(f) {
  const d = el('div', 'field');
  const isTopo = f.k === '__topology';
  const val = isTopo ? (S.target?.topology || '') : (S.params[f.k] ?? '');
  if (isTopo && !val) d.classList.add('miss');
  const lab = el('label', null, esc(f.label));
  const q = el('span', 'q', 'ⓘ'); q.title = f.q; lab.appendChild(q);
  d.appendChild(lab);

  let inp;
  if (f.type === 'topo') {
    inp = el('select');
    inp.appendChild(el('option', null, '— 请选择 —')).value = '';
    Object.entries(TOPOS).forEach(([k, v]) => {
      const o = el('option', null, `${k} · ${v.chips} 芯片`); o.value = k;
      if (k === val) o.selected = true; inp.appendChild(o);
    });
  } else if (f.type === 'sel') {
    inp = el('select');
    f.opts.forEach(o => { const x = el('option', null, o || '— 不设 —'); x.value = o;
      if (String(val) === o) x.selected = true; inp.appendChild(x); });
  } else if (f.type === 'bool') {
    inp = el('select');
    [['', '— 不设 —'], ['True', 'True'], ['False', 'False']].forEach(([v, t]) => {
      const x = el('option', null, t); x.value = v;
      if (String(val) === v || (val === true && v === 'True') || (val === false && v === 'False')) x.selected = true;
      inp.appendChild(x);
    });
  } else {
    inp = el('input');
    inp.type = f.type === 'num' ? 'number' : 'text';
    inp.value = val;
  }
  inp.onchange = () => setParam(f.k, inp.value);
  d.appendChild(inp);
  return d;
}

/* ── 渲染：命令 ───────────────────────────────────────────── */
function renderCmd() {
  const p = $('#pane-cmd'); p.innerHTML = '';
  const c = el('div', 'card');
  const h = el('h3', null, 'AOT 命令');
  const cp = el('button', 'btn-ghost btn-sm', '复制');
  cp.style.marginLeft = 'auto';
  cp.onclick = () => { navigator.clipboard.writeText(S.aot_cmd); toast('已复制'); };
  h.appendChild(cp); c.appendChild(h);
  c.appendChild(el('div', 'hint', '在 CPU 上跑，不占加速卡。绿底是 tpuguru 替你加的。'));
  const hi = esc(S.aot_cmd).replace(/^(\s*)([\w.]+)=(.*)$/gm, (m, sp, k, v) =>
    `${sp}<span class="${S.added && k in S.added ? 'add' : ''}"><span class="k">${k}</span>=<span class="v">${v}</span></span>`);
  c.appendChild(el('pre', 'cmd', hi));
  p.appendChild(c);

  if (S.dropped && Object.keys(S.dropped).length) {
    const d = el('div', 'card');
    d.appendChild(el('h3', null, '已丢弃的运行时参数'));
    d.appendChild(el('div', 'hint', 'AOT 只编译不训练，这些留着反而会干扰或报错。'));
    d.appendChild(el('table', 't', '<tbody>' + Object.entries(S.dropped).map(([k, v]) =>
      `<tr><td class="mono">${esc(k)}</td><td class="mono" style="color:var(--faint)">${esc(v)}</td></tr>`).join('') + '</tbody>'));
    p.appendChild(d);
  }

  const rt = el('div', 'card');
  rt.appendChild(el('h3', null, '回环校验'));
  if (!S.roundtrip.length) {
    rt.appendChild(el('div', 'hint',
      '✅ 生成的命令再解析一遍，跟表单逐项一致。<br>这一步是为了保证你看到的命令就是会被执行的命令 —— 不静默采纳任何改写。'));
  } else {
    rt.appendChild(el('div', 'hint', '⚠️ 有对不上的项，先确认再跑：'));
    rt.appendChild(el('table', 't', '<thead><tr><th>参数</th><th>表单</th><th>命令里</th></tr></thead><tbody>'
      + S.roundtrip.map(d => `<tr><td class="mono">${esc(d.param)}</td><td class="mono">${esc(d.expected)}</td><td class="mono">${esc(d.actual)}</td></tr>`).join('')
      + '</tbody>'));
  }
  p.appendChild(rt);

  if (S.raw_cmd) {
    const o = el('div', 'card');
    o.appendChild(el('h3', null, '原始命令'));
    o.appendChild(el('pre', 'cmd', esc(S.raw_cmd)));
    p.appendChild(o);
  }
}

/* ── 渲染：报告 ───────────────────────────────────────────── */
const SEG_COLORS = ['#1a73e8', '#00acc1', '#34a853', '#f9ab00'];

function renderRep() {
  const p = $('#pane-rep'); p.innerHTML = '';
  const R = S.result;
  if (!R) {
    // ★ 报告是按**当前配置**索引的。这套配置没跑过就空着 ——
    //   显示上一套配置的报告，等于让人对着 A 的结论调 B 的参数。
    const seen = everSawReport || (S.known_fingerprints || []).length || (S.run_ids || []).length;
    const e = el('div', 'empty',
      seen
        ? '<span class="ic">📊</span><b>这套配置还没跑过</b><br>'
          + '报告是按配置索引的 —— 改回之前跑过的配置，那份报告会自己回来。'
        : '<span class="ic">📊</span>还没跑过 AOT<br>大约 3 分钟，不占加速卡');
    const btn = el('div', 'cta', '跑 AOT');
    btn.onclick = () => $('#btn-run').click();
    e.appendChild(btn);
    e.appendChild(el('span', 'fp', '配置指纹 ' + esc(S.fingerprint || '')));
    p.appendChild(e);
    return;
  }
  everSawReport = true;
  if (S.cached_from) {
    p.appendChild(el('div', 'cachenote',
      '<div>🗂</div><div>这份报告是<b>之前跑过的同一套配置</b>直接调出来的'
      + `（${esc(S.cached_from.at || '')}），没有重跑。`
      + '配置只要有一个字不同，指纹就变，这里就会空掉。</div>'));
  }

  // ☠️ 作废的数字必须先于一切出现，并且把它自己划掉。
  // 「编译通过 + 一个漂亮数字」是这份报告最危险的呈现方式。
  if (R.invalid) {
    const vb = el('div', 'void-banner');
    vb.appendChild(el('div', 't', '⛔ 这一档的数字已作废，不要引用'));
    vb.appendChild(el('div', 'd', md2(R.invalid)));
    p.appendChild(vb);
  }
  if (R.nonmono) {
    p.appendChild(el('div', 'stale', '<div>📉</div><div>' + md2(R.nonmono) + '</div>'));
  }

  if (R.projection) {
    const P = R.projection;
    const bn = el('div', 'projbanner');
    bn.appendChild(el('div', 't', `📐 推算：${P.from_layers} 层 → ${P.to_layers} 层（比例 ${P.ratio}）`));
    bn.appendChild(el('div', 'd', md2(P.why) + '<br>' + md2(P.caveat)));
    bn.appendChild(el('div', 'ref', `参照点：${fmt(P.ref_peak_gb)} GB · ${esc(P.ref_source)}`));
    if (P.decompose) bn.appendChild(el('div', 'd',
      '<br>' + md2(P.decompose.why)));
    if (P.recommend) {
      const r = P.recommend;
      const box = el('div', 'reco');
      box.appendChild(el('div', 'big2', `推荐 batch ${r.safe}`
        + `<span class="sub">硬上限约 ${r.hard}</span>`));
      box.appendChild(el('div', 'fx', esc(r.formula)));
      box.appendChild(el('div', 'd', md2(r.note)));
      const chips = el('div', 'plan');
      chips.appendChild(el('span', 'pl', '建议逐档真跑：'));
      r.plan.forEach(v => { const cbtn = el('span', 'pchip', 'pdbs ' + v);
        cbtn.onclick = () => setParam('per_device_batch_size', String(v));
        chips.appendChild(cbtn); });
      box.appendChild(chips);
      bn.appendChild(box);
    }
    p.appendChild(bn);
  }

  // 结论
  const v = el('div', 'card');
  const ok = R.ok;
  const pj = R.mode === 'projected';
  const txt = R.invalid ? '编译通过，但结果无效'
    : ok === true ? (pj ? '📐 推算：装得下' : '✅ 编译通过，装得下')
    : ok === false ? (pj ? '📐 推算：装不下' : '❌ 装不下') : '❓ 这一档没有实测记录';
  v.appendChild(el('div', 'big ' + (R.invalid ? 'void' : ok === true ? 'ok' : ok === false ? 'bad' : 'unk'), txt));
  if (R.failure) {
    v.appendChild(el('div', 'hint', `${esc(R.failure.kind)} · 需要 <b>${fmt(R.failure.required_gb)} GB</b>，`
      + `上限 ${fmt(R.failure.available_gb)} GB，超 <b>${fmt(R.failure.required_gb - R.failure.available_gb)} GB</b>`));
  }
  if (R.note) v.appendChild(el('div', 'hint', esc(R.note)));
  const m = R.metrics || {};
  if (Object.keys(m).length) {
    const k = el('div', 'kpis');
    const kp = (n, l, cls) => { const d = el('div', 'kpi' + (cls ? ' ' + cls : ''));
      d.appendChild(el('div', 'n', n)); d.appendChild(el('div', 'l', l)); return d; };
    if (m.peak_hbm_gb) k.appendChild(kp(fmt(m.peak_hbm_gb) + ' <span style="font-size:12px">GB</span>', 'HBM 峰值 / device'));
    if (m.hbm_pct) k.appendChild(kp(m.hbm_pct + '%', '容量占用',
      m.hbm_pct > 100 ? 'over' : m.hbm_pct > 95 ? 'hot' : ''));
    if (m.global_batch) k.appendChild(kp(m.global_batch.toLocaleString(), 'global batch（条）'));
    if (m.fsdp) k.appendChild(kp(m.fsdp, 'FSDP 宽度'));
    if (m.end_to_end_s) k.appendChild(kp(m.end_to_end_s + ' s', 'AOT 耗时'));
    if (R.invalid) k.querySelectorAll('.kpi').forEach(x => x.classList.add('struck'));
    v.appendChild(k);
  }
  const badge = el('div', 'hint');
  badge.style.marginTop = '12px';
  badge.innerHTML = R.mode === 'projected'
    ? `<span class="pill warn"><span class="dot"></span>推算模式</span> `
      + `按层数从实测点折算，<b>不是这个层数的实测</b>。要定上限请设 AOT 镜像后真跑。`
    : R.mode === 'replay'
    ? `<span class="pill warn"><span class="dot"></span>replay 模式</span> 本机没有 AOT 镜像，回放的是真实跑过的结论。`
      + (R.source ? `<br>出处：${esc(R.source)}` : '')
    : `<span class="pill"><span class="dot"></span>real 模式</span> 本机 docker 真跑的。`;
  v.appendChild(badge);
  p.appendChild(v);

  const m2 = R.metrics || {};
  if (m2.per_chip_tflops || m2.step_s) p.appendChild(cardRealRef(m2, R));

  const A = R.analyses || {};
  if (A.scale) p.appendChild(cardScale(A.scale.data));
  if (A.memory) p.appendChild(cardMemory(A.memory.data));
  if (A.headroom && A.headroom.data.ready) p.appendChild(cardHeadroom(A.headroom.data));
  if (A.levers && A.levers.data.ready) p.appendChild(cardLevers(A.levers.data));
  if (A.codepath) p.appendChild(cardCodepath(A.codepath.data));
  if (A.collectives) p.appendChild(cardCollectives(A.collectives.data));
  if (A.compile_time) p.appendChild(cardCompile(A.compile_time.data));
  if (A.hlo) p.appendChild(cardHlo(A.hlo.data));
  if (A.llo) p.appendChild(cardLlo(A.llo.data));
  if (R.artifacts && Object.keys(R.artifacts).length) p.appendChild(cardArtifacts(R.artifacts));
}

function cardLlo(d) {
  const c = el('div', 'card');
  c.appendChild(el('h3', null, 'LLO（低层指令）'));
  if (d.collected) { c.appendChild(el('div', 'hint', '已采集')); return c; }
  c.appendChild(el('div', 'notyet', esc(d.why) + '<br><br>' + esc(d.howto)));
  return c;
}

const KB = n => n < 1024 ? n + ' B' : n < 1048576 ? (n / 1024).toFixed(0) + ' KB'
  : (n / 1048576).toFixed(1) + ' MB';

function cardArtifacts(a) {
  const c = el('div', 'card');
  c.appendChild(el('h3', null, '编译产物'));
  c.appendChild(el('div', 'hint',
    '存档时这些会被复制到存档专区并去掉生命周期 —— 临时区挂着 30 天清理规则，'
    + '<b>只记一个临时路径的存档，一个月后点开是 404 且不报错</b>。'));
  const g = el('div', 'arts');
  Object.entries(a).forEach(([k, v]) => {
    const d = el('div', 'art');
    const kind = v.kind === 'hlo' ? 'hlo' : v.kind === 'json' ? 'json' : 'log';
    d.appendChild(el('div', 'ic ' + kind, kind.toUpperCase()));
    const t = el('div'); t.style.minWidth = '0';
    const short = v.name.length > 30 ? v.name.slice(0, 14) + '…' + v.name.slice(-13) : v.name;
    const nm = el('div', 'nm', esc(short)); nm.title = v.name;
    t.appendChild(nm);
    t.appendChild(el('div', 'sz', KB(v.bytes)));
    if (v.desc) t.appendChild(el('div', 'ds', esc(v.desc)));
    d.appendChild(t); g.appendChild(d);
  });
  c.appendChild(g);
  return c;
}

/* AOT 不产生吞吐数字。这里显示的是**同配置真机跑过的记录**，
   跟 AOT 的结论必须在视觉上分开，否则会被当成 AOT 预测出来的。 */
function cardRealRef(m, R) {
  const c = el('div', 'card realref');
  const h = el('h3', null, '真机参照');
  h.appendChild(el('span', 'tagline', '不是 AOT 算的'));
  c.appendChild(h);
  c.appendChild(el('div', 'hint', md2(
    'AOT 只编译不执行，**它给不出任何吞吐数字**。下面是同一套配置在真机上跑过的记录，'
    + '放在这里是为了让你把「装不装得下」和「快不快」对上号。')));
  const k = el('div', 'kpis');
  const kp = (n, l, cls) => { const d = el('div', 'kpi' + (cls ? ' ' + cls : ''));
    d.appendChild(el('div', 'n', n)); d.appendChild(el('div', 'l', l)); return d; };
  if (m.per_chip_tflops) k.appendChild(kp(fmt(m.per_chip_tflops, 1),
    'TFLOP/s/chip', R.invalid ? 'struck' : ''));
  if (m.per_chip_tflops) k.appendChild(kp((m.per_chip_tflops / 2307 * 100).toFixed(1) + '%',
    'MFU（BF16 峰值 2307）', R.invalid ? 'struck' : ''));
  if (m.step_s) k.appendChild(kp(fmt(m.step_s, 2) + ' s', 'step 时间', R.invalid ? 'struck' : ''));
  c.appendChild(k);
  if (R.invalid) c.appendChild(el('div', 'hint',
    '<b style="color:var(--red)">上面这几个数已作废</b>，划掉的原因见页首。'));
  return c;
}

function cardScale(d) {
  const c = el('div', 'card');
  c.appendChild(el('h3', null, '模型与规模'));
  const g = el('div', 'shape');
  d.items.forEach(i => {
    const b = el('div', i.v === '—' ? 'na' : null);
    b.appendChild(el('div', 'v', esc(i.v)));
    b.appendChild(el('div', 'l', esc(i.l)));
    b.title = { lookup: '来自模型形状表', config: '你配置里写的', calc: '由配置精确算出' }[i.kind] || '';
    g.appendChild(b);
  });
  c.appendChild(g);
  if (d.note) c.appendChild(el('div', 'hint', md(d.note).replace(/<\/?p>/g, '')));
  return c;
}

function cardMemory(d) {
  const c = el('div', 'card');
  c.appendChild(el('h3', null, 'HBM 分解 · 每 device'));
  c.appendChild(el('div', 'hint',
    '常驻部分按参数量精确算；<b>激活是由峰值倒推的</b>，不是独立测得 —— 标了「推算」的别当实测引。'));
  const cap = d.capacity_gb, peak = d.peak_gb;
  const box = el('div', 'hbm');
  const scale = Math.max(cap, peak || 0) * 1.06;
  const wrap = el('div', 'wrap');
  const track = el('div', 'track');
  (d.segments || []).forEach((s, i) => {
    const w = (s.gb / scale) * 100;
    const seg = el('div', `seg s${i}` + (s.kind === 'derived' ? ' derived' : ''));
    seg.style.width = w + '%';
    seg.title = `${s.name} ${fmt(s.gb)} GB${s.note ? ' · ' + s.note : ''}`;
    track.appendChild(seg);
  });
  if (d.over_gb > 0) {
    const o = el('div', 'seg over'); o.style.width = (d.over_gb / scale) * 100 + '%';
    o.title = `超出 ${fmt(d.over_gb)} GB`; track.appendChild(o);
  }
  wrap.appendChild(track);
  const capm = el('div', 'cap', `<span>上限 ${fmt(cap)} GB</span>`);
  capm.style.left = (cap / scale) * 100 + '%';
  wrap.appendChild(capm);
  box.appendChild(wrap);

  const lg = el('div', 'legend');
  (d.segments || []).forEach((s, i) => {
    lg.appendChild(el('span', null,
      `<i style="background:${SEG_COLORS[i % 4]}"></i>${esc(s.name)} `
      + `<span class="g">${fmt(s.gb)} GB</span>`
      + (s.kind === 'derived' ? '<span class="est">推算</span>' : '')));
  });
  if (d.over_gb > 0) lg.appendChild(el('span', null,
    `<i style="background:#d93025"></i>超出 <span class="g">${fmt(d.over_gb)} GB</span>`));
  box.appendChild(lg);
  c.appendChild(box);
  if (peak) {
    const room = cap - peak;
    c.appendChild(el('div', 'hint', room >= 0
      ? `离上限还剩 <b>${fmt(room)} GB</b>（${(room / cap * 100).toFixed(1)}%）。`
        + '余量小于 2 GB 时不要直接上真机 —— 显存不随 batch 单调，换个尺寸编译器可能换排布方案。'
      : `超出 <b>${fmt(-room)} GB</b>。先加宽 FSDP，那是最有效的杠杆；改重算策略排在它后面。`));
  }
  return c;
}

/* ★ 「batch 还能开多大」—— 拆成三个能看懂的数，再配一张真实档位阶梯。 */
function cardHeadroom(d) {
  const c = el('div', 'card');
  c.appendChild(el('h3', null, '显存余量 · batch 还能开多大'));
  if (d.verdict) {
    const v = el('div', 'verdict', md2(d.verdict));
    c.appendChild(v);
  }

  const g = el('div', 'split3');
  const cell = (n, cls, l, sub) => { const x = el('div');
    x.appendChild(el('div', 'n ' + cls, n)); x.appendChild(el('div', 'l', l));
    if (sub) x.appendChild(el('div', 's', sub)); return x; };
  g.appendChild(cell(fmt(d.resident_gb) + ' <span style="font-size:12px">GB</span>', 'blue',
    '常驻（不随 batch 变）', '主权重 + 优化器状态，只跟 FSDP 宽度有关'));
  g.appendChild(cell(fmt(d.batch_gb) + ' <span style="font-size:12px">GB</span>', 'amber',
    `随 batch 走（当前 pdbs ${d.pdbs}）`, '激活 + 临时缓冲'));
  const lc = d.left_gb === null ? 'green' : d.left_gb < 0 ? 'red' : d.left_gb < 2 ? 'amber' : 'green';
  g.appendChild(cell((d.left_gb >= 0 ? '' : '超 ') + fmt(Math.abs(d.left_gb))
    + ' <span style="font-size:12px">GB</span>', lc,
    d.left_gb >= 0 ? '离上限还剩' : '超出上限', `上限 ${fmt(d.capacity_gb)} GB / device`));
  c.appendChild(g);

  if (d.solved) {
    c.appendChild(el('div', 'fxline',
      `<b>${esc(d.solved.formula)}</b><br>`
      + `反解：保守 <b>${d.solved.safe}</b> / 硬上限 <b>${d.solved.hard}</b>。`
      + '「固定激活」是**跟 batch 无关**的那部分（梯度缓冲、通信缓冲、编译器工作区）——'
      + '不把它拆出来，直接拿「非常驻 ÷ batch」算，上限会大出一倍。'
      .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')));
  }
  if (d.per_batch_gb) {
    const kindTxt = d.slope_kind === 'measured' ? '由相邻实测档位算出' : '估算（偏大，只能当上界）';
    c.appendChild(el('div', 'hint',
      `<b>batch 每 +1 档 ≈ ${fmt(d.per_batch_gb)} GB</b>（${kindTxt}）。`
      + md2(d.more_text || '')
      + (d.slope_note ? '<br>' + md2(d.slope_note) : '')));
  }

  if (d.ladder && d.ladder.length) {
    const scale = Math.max(d.capacity_gb, ...d.ladder.map(x => x.gb)) * 1.05;
    const box = el('div', 'ladder');
    d.ladder.forEach(x => {
      const r = el('div', 'lrow' + (x.ok ? '' : ' bad') + (x.cur ? ' cur' : ''));
      r.appendChild(el('div', 'lb', 'pdbs ' + x.pdbs));
      const t = el('div', 'lt'); const f = el('div', 'lf');
      f.style.width = Math.min(x.gb / scale * 100, 100) + '%'; t.appendChild(f); r.appendChild(t);
      r.appendChild(el('div', 'lv', fmt(x.gb) + ' GB' + (x.ok ? '' : ' ✕')));
      box.appendChild(r);
    });
    const cap = el('div', 'capline', `<span>上限 ${fmt(d.capacity_gb)}</span>`);
    cap.style.left = `calc(62px + 10px + (100% - 62px - 96px - 20px) * ${d.capacity_gb / scale})`;
    box.appendChild(cap);
    c.appendChild(box);
    c.appendChild(el('div', 'hint', '<br>灰底是实测档位，✕ 是装不下的。' + md2(d.warn)));
  }
  return c;
}

/* ★ 「该拧哪个旋钮」—— 按能腾出多少 GB 排序，直接换算成等价 batch 档数。 */
function cardLevers(d) {
  const c = el('div', 'card');
  c.appendChild(el('h3', null, '该拧哪个旋钮'));
  c.appendChild(el('div', 'hint',
    '按**能腾出多少显存**排序，并换算成「等价几档 batch」—— 这样能直接比较「加宽 FSDP」和「降 batch」哪个更划算。'
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')));
  d.rows.forEach(r => {
    const x = el('div', 'lever' + (r.forbidden ? ' forbidden' : ''));
    const L = el('div');
    L.appendChild(el('div', 'nm', esc(r.name)));
    L.appendChild(el('div', 'why', md2(r.why)));
    if (r.how) L.appendChild(el('div', 'how', esc(r.how)));
    if (r.risk && r.risk !== '无') L.appendChild(el('div', 'why',
      '<span style="color:#8a6d00">风险：</span>' + md2(r.risk)));
    x.appendChild(L);
    const R = el('div', 'amt');
    R.appendChild(el('div', 'g', '+' + fmt(r.gb) + ' GB'));
    if (r.eq_batch) R.appendChild(el('div', 'e', `≈ ${r.eq_batch} 档 batch`));
    x.appendChild(R);
    c.appendChild(x);
  });
  return c;
}

function cardCodepath(d) {
  const c = el('div', 'card');
  c.appendChild(el('h3', null, '★ 实际走到了哪条代码路径'));
  c.appendChild(el('div', 'hint',
    '这是 tpuguru 存在的主要理由 —— 配置在语义上出错却不报错，只能靠把实际分支打出来发现。'));
  const row = el('div', 'path');
  row.appendChild(el('span', 'chip', esc(d.branch)));
  row.appendChild(el('span', 'arrow', '→'));
  row.appendChild(el('span', 'chip ' + (d.weight_gather.startsWith('❌') ? 'bad' : 'good'),
    '权重收集：' + esc(d.weight_gather)));
  row.appendChild(el('span', 'chip mute', '专家维分片：' + (d.shard_expert_dim ? '开' : '关')));
  c.appendChild(row);
  if (d.risk) c.appendChild(el('div', 'danger', '<b>⚠️ 静默出错风险</b><br>' + esc(d.risk)));
  c.appendChild(el('div', 'hint', '<br>怎么自己验：' + esc(d.probe)));
  return c;
}

function cardCollectives(d) {
  const c = el('div', 'card');
  c.appendChild(el('h3', null, '集合通信'));
  c.appendChild(el('div', 'hint', '<b>看执行次数，别只看字节数。</b>每步一次 = 编译器把各层合并、提升出了循环；每层一次 = 手写的，提不出去。'));
  const rows = (d.rows || []).map(r => `<tr>
    <td>${esc(r.op)}</td>
    <td class="mono">${r.per_step}</td>
    <td><span class="tag ${r.hoisted ? 'ok' : 'no'}">${r.hoisted ? '已提出循环' : '钉在循环里'}</span></td>
    <td class="mono">${r.exposed_ms === null ? '—' : fmt(r.exposed_ms, 1) + ' ms'}</td>
    <td style="color:var(--muted)">${esc(r.note)}</td></tr>`).join('');
  c.appendChild(el('table', 't',
    '<thead><tr><th>算子</th><th>次数/步</th><th>调度</th><th>暴露耗时</th><th></th></tr></thead><tbody>' + rows + '</tbody>'));
  if (d.insight) c.appendChild(el('div', 'hint', '<br>' + esc(d.insight)));
  return c;
}

function cardCompile(d) {
  const c = el('div', 'card');
  c.appendChild(el('h3', null, `编译时间 · 合计 ${d.total_s} s`));
  c.appendChild(el('div', 'hint', '真机训练时这段时间要再花一遍。产物缓存下来就能省掉。'));
  const max = Math.max(...d.phases.map(p => p.s));
  const bars = el('div', 'bars');
  d.phases.forEach(p => {
    const b = el('div', 'bar');
    b.appendChild(el('div', 'nm', esc(p.name)));
    const tr = el('div', 'tr'); const fl = el('div', 'fl');
    fl.style.width = (p.s / max * 100) + '%'; tr.appendChild(fl); b.appendChild(tr);
    b.appendChild(el('div', 'vl', p.s + ' s'));
    bars.appendChild(b);
  });
  c.appendChild(bars);
  return c;
}

function cardHlo(d) {
  const c = el('div', 'card');
  c.appendChild(el('h3', null, 'HLO 统计'));
  c.appendChild(el('div', 'hint',
    `${d.instructions.toLocaleString()} 条指令、${d.fusions.toLocaleString()} 个 fusion。`
    + '下面按静态成本占比排 —— <b>这是编译器的估算，不是实测时间</b>。'));
  const max = Math.max(...d.top.map(t => t.pct));
  const bars = el('div', 'bars');
  d.top.forEach(t => {
    const b = el('div', 'bar');
    b.appendChild(el('div', 'nm', `<span class="mono" style="font-size:12px">${esc(t.op)}</span>`
      + (t.note ? `<small>${esc(t.note)}</small>` : '')));
    const tr = el('div', 'tr'); const fl = el('div', 'fl');
    fl.style.width = (t.pct / max * 100) + '%'; tr.appendChild(fl); b.appendChild(tr);
    b.appendChild(el('div', 'vl', t.pct + '%'));
    bars.appendChild(b);
  });
  c.appendChild(bars);
  return c;
}

/* ── 渲染：历史 ───────────────────────────────────────────── */
async function renderHis() {
  const p = $('#pane-his'); p.innerHTML = '<div class="hint">载入中…</div>';
  let list = [];
  try { list = (await api('/api/saves')).saves; } catch (e) { p.innerHTML = ''; }
  p.innerHTML = '';
  const c = el('div', 'card');
  c.appendChild(el('h3', null, '存档'));
  c.appendChild(el('div', 'hint',
    '存档冻住的是整个现场：配置、对话、体检结论、分析报告、日志与编译产物。'
    + '<b>载入 = 派生新会话</b>，原存档只读。作废的不删，留着记「这条路走不通」。'));
  if (!list.length) {
    c.appendChild(el('div', 'empty', '<span class="ic">💾</span>还没有存档<br>调到满意时点右上角【存档】'));
    p.appendChild(c); return;
  }
  const nVoid = list.filter(x => x.voided).length;
  const bar = el('div', 'listbar');
  bar.appendChild(el('span', null, `${list.length} 个存档` + (nVoid ? ` · ${nVoid} 个已作废` : '')));
  bar.appendChild(el('span', 'spacer'));
  if (nVoid) {
    const lb = el('label');
    const cb = el('input'); cb.type = 'checkbox'; cb.checked = hideVoided;
    cb.onchange = () => { hideVoided = cb.checked; renderHis(); };
    lb.append(cb, document.createTextNode('隐藏已作废'));
    bar.appendChild(lb);
  }
  c.appendChild(bar);
  if (hideVoided) list = list.filter(x => !x.voided);
  // 按 parent_save_id 摆成树。**只有分叉才缩进** ——
  // 线性调优（每次在上一版基础上改一点）是一条链，逐级缩进会越缩越偏。
  const byId = Object.fromEntries(list.map(x => [x.id, x]));
  const kids = {}; const roots = [];
  list.forEach(x => {
    if (x.parent_save_id && byId[x.parent_save_id]) (kids[x.parent_save_id] ||= []).push(x);
    else roots.push(x);
  });
  const byTime = (a, b) => (a.created_at || '').localeCompare(b.created_at || '');
  const tree = el('div', 'tree');

  function chain(start, container) {
    const lane = el('div', 'lane');
    let cur = start;
    while (cur) {
      lane.appendChild(nodeOf(cur));
      const ch = (kids[cur.id] || []).sort(byTime);
      if (ch.length === 1) { cur = ch[0]; continue; }      // 继续同一条链
      if (ch.length > 1) {                                  // 分叉才缩进
        const fk = el('div', 'fork');
        fk.appendChild(el('div', 'lanehead', `↳ 从这里分出 ${ch.length} 条支线`));
        ch.forEach(k => chain(k, fk));
        lane.appendChild(fk);
      }
      cur = null;
    }
    container.appendChild(lane);
  }
  roots.sort((a, b) => byTime(b, a)).forEach(r => chain(r, tree));
  c.appendChild(tree);
  p.appendChild(c);
}

function nodeOf(s, depth) {
  const n = el('div', 'node depth-' + depth + (s.voided ? ' void' : ''));
  const left = el('div');
  left.appendChild(el('div', 'ttl', (s.voided ? '⛔ ' : '💾 ') + esc(s.title || '(未命名)')));
  const rs = s.result_summary || {}; const m = s.metrics || {};
  const bits = [s.created_at || ''];
  if (m.peak_hbm_gb) bits.push(fmt(m.peak_hbm_gb) + ' GB');
  if (m.pdbs) bits.push('pdbs ' + m.pdbs);
  if (rs.per_chip_tflops) bits.push(fmt(rs.per_chip_tflops, 1) + ' TFLOP/s');
  // 作废的不打勾 —— 它编译是通过了，但结论无效，打勾会让人误以为可用
  if (s.voided || rs.invalid) bits.push('⛔ 结论无效');
  else if (rs.ok === true) bits.push('✅');
  else if (rs.ok === false) bits.push('❌');
  left.appendChild(el('div', 'meta', esc(bits.join('  ·  '))));
  if (s.note) left.appendChild(el('div', 'meta', esc(s.note)));
  if (s.tags && s.tags.length) left.appendChild(el('div', 'meta',
    s.tags.map(t => `<span class="tag ok" style="margin-right:5px">${esc(t)}</span>`).join('')));
  if (s.voided) {
    const r = String(s.voided.reason || '').replace(/\*\*/g, '');
    left.appendChild(el('div', 'meta',
      (s.voided.by === 'auto' ? '自动作废：' : '作废原因：') + esc(r.length > 110 ? r.slice(0, 110) + '…' : r)));
  }
  n.appendChild(left);

  const acts = el('div', 'acts');
  const load = el('button', 'btn-ghost btn-sm', '载入');
  load.onclick = async (e) => {
    e.stopPropagation();
    if (!confirm(`载入「${s.title}」？会开一场新对话，当前会话保留。`)) return;
    setBusy(true);
    try { S = await api(`/api/save/${s.id}/load`, {}); switchTab('cfg'); render(); toast('已载入'); }
    catch (err) { toast('载入失败：' + err.message); } finally { setBusy(false); }
  };
  acts.appendChild(load);
  if (!s.voided) {
    const vd = el('button', 'btn-ghost btn-sm', '作废');
    vd.onclick = async (e) => {
      e.stopPropagation();
      const r = prompt(`作废「${s.title}」\n\n原因（会一起存下来，以后能看到「这条路为什么走不通」）：`);
      if (!r) return;
      await api(`/api/save/${s.id}/void`, { reason: r }); renderHis(); toast('已标记作废');
    };
    acts.appendChild(vd);
  }
  n.appendChild(acts);
  return n;
}

/* ── 主渲染 ───────────────────────────────────────────────── */
function render() {
  renderTurns(); renderCfg(); renderCmd(); renderRep();
  $('#sess-title').textContent = S.session_id;
  const fatal = S.lint.filter(f => f.severity === 'fatal').length;
  const warn = S.lint.filter(f => f.severity === 'warn').length;
  const b = $('#b-lint');
  if (fatal || warn) { b.hidden = false; b.textContent = fatal || warn;
    b.className = 'badge ' + (fatal ? 'red' : 'amber'); } else b.hidden = true;
  const br = $('#b-rep');
  if (S.result) {
    br.hidden = false;
    br.textContent = S.result.invalid ? '⛔' : S.result.ok === false ? '!' : '✓';
    br.className = 'badge ' + (S.result.invalid || S.result.ok === false ? 'red'
      : S.result.ok ? 'green' : '');
    $('.tab[data-pane="rep"]').title = '这份报告对应当前配置';
  } else {
    br.hidden = true; br.textContent = '';
    $('.tab[data-pane="rep"]').title = '当前配置还没跑过 AOT';
  }
  // 顶栏即时指示：不用切到报告页就知道这套配置有没有结论
  const fs = $('#fpstate');
  const has = !!S.result;
  fs.className = 'fpstate ' + (has ? 'has' : 'none');
  fs.lastElementChild.textContent = has
    ? (S.cached_from ? '已有报告（调档）' : '已有报告') : '未跑 AOT';
  fs.title = '配置指纹 ' + (S.fingerprint || '');

  // 🚀 上机按钮：只有「当前配置的 AOT 编译通过」才点亮。
  //    真机一次几十分钟 + 占共享集群，装不下就上机纯属浪费别人的卡。
  const bm = $('#btn-metal');
  const compiled = !!(S.result && S.result.ok === true && !S.result.invalid);
  const clOk = CLUSTER && CLUSTER.light !== 'red' && CLUSTER.ok;
  bm.disabled = !(compiled && clOk);
  bm.title = !compiled
    ? (S.result ? 'AOT 说装不下 / 结论无效，先把它跑通' : '这套配置还没跑过 AOT')
    : !clOk ? '集群现在要不到卡：' + (CLUSTER?.text || '')
    : `上 64 卡真跑（约 20–40 分钟，占共享集群）\n峰值 ${S.result.metrics?.peak_hbm_gb} GB，已编译通过`;
  if ($('.tab.on')?.dataset.pane === 'his') renderHis();
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('on', t.dataset.pane === name));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('on', p.id === 'pane-' + name));
  if (name === 'his') renderHis();
}

/* ── 事件 ─────────────────────────────────────────────────── */
document.querySelectorAll('.tab').forEach(t => t.onclick = () => switchTab(t.dataset.pane));

async function send() {
  const t = $('#input').value.trim();
  if (!t || busy) return;
  $('#input').value = ''; $('#input').style.height = 'auto';
  setBusy(true);
  // 乐观插入，别让人对着静止的界面等
  S.turns.push({ role: 'user', text: t }); renderTurns();
  try { S = await api('/api/chat', { session_id: S.session_id, text: t }); render(); }
  catch (e) { toast('出错：' + e.message); }
  finally { setBusy(false); $('#input').focus(); }
}
$('#btn-send').onclick = send;
$('#input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
$('#input').addEventListener('input', e => {
  e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px';
});

$('#btn-run').onclick = async () => {
  if (busy) return;
  setBusy(true, '编译中…');
  try { S = await api('/api/run', { session_id: S.session_id, text: '' }); switchTab('rep'); render(); }
  catch (e) { toast(e.message); }
  finally { setBusy(false); }
};

$('#btn-new').onclick = async () => {
  S = await api('/api/session', {}); switchTab('cfg'); render(); toast('新会话');
};

$('#btn-save').onclick = () => {
  const mask = el('div', 'mask');
  const m = el('div', 'modal');
  m.innerHTML = `<h3>💾 存档</h3>
    <p class="hint">把当前配置、对话、体检结论、分析报告和产物冻成一份不可变快照。
      以后从历史里载入，会开一场新对话 —— 存档本身不会被改。</p>
    <input id="sv-t" placeholder="标题，例如：v3 去掉 QAG，FSDP 128 → 727.0">
    <textarea id="sv-n" rows="2" placeholder="备注（可选）：这套是干什么用的、有什么前提"></textarea>
    <input id="sv-g" placeholder="标签，逗号分隔（可选）">
    <div class="row"><button class="btn-ghost" id="sv-c">取消</button>
      <button class="btn-primary" id="sv-ok">存档</button></div>`;
  mask.appendChild(m); document.body.appendChild(mask);
  const close = () => mask.remove();
  mask.onclick = e => { if (e.target === mask) close(); };
  m.querySelector('#sv-c').onclick = close;
  m.querySelector('#sv-t').focus();
  m.querySelector('#sv-ok').onclick = async () => {
    const title = m.querySelector('#sv-t').value.trim();
    if (!title) { toast('给它起个名字'); return; }
    try {
      const r = await api('/api/save', { session_id: S.session_id, title,
        note: m.querySelector('#sv-n').value.trim(),
        tags: m.querySelector('#sv-g').value.split(',').map(x => x.trim()).filter(Boolean) });
      S = r.state; close(); switchTab('his'); render(); toast('已存档');
    } catch (e) { toast('存档失败：' + e.message); }
  };
};

async function refreshCluster() {
  try {
    CLUSTER = await api('api/cluster?want=64');
  } catch (e) { CLUSTER = { ok: false, light: 'grey', text: '探测失败', why: String(e) }; }
  const n = $('#cluster');
  n.className = 'cl ' + (CLUSTER.light || 'grey');
  n.lastElementChild.innerHTML = esc(CLUSTER.text || '')
    + (CLUSTER.ok ? ` <span class="n">空闲 ${CLUSTER.cohort_free}/${CLUSTER.cohort_total}</span>` : '');
  n.title = (CLUSTER.why || '') + (CLUSTER.ok
    ? `\n保底 ${CLUSTER.quota} · 我们已用 ${CLUSTER.used} · Running pod ${CLUSTER.running_pods}`
    : '');
  if (S) render();
}

/* ── 启动 ─────────────────────────────────────────────────── */
(async () => {
  try {
    const h = await api('/api/health');
    TOPOS = h.topologies || {};
    FAMILIES = h.families || []; BACKENDS = h.backends || [];
    const ps = $('#pill-store');
    ps.className = 'pill' + (h.store === 'firestore' ? '' : ' warn');
    ps.lastElementChild.textContent = h.store === 'firestore' ? 'Firestore' : '本地存储';
    const pm = $('#pill-mode');
    pm.className = 'pill' + (h.aot_mode === 'real' ? '' : ' warn');
    pm.lastElementChild.textContent = h.aot_mode === 'real' ? 'AOT: real' : 'AOT: replay';
    pm.title = h.aot_mode === 'real' ? '本机 docker 真跑' : '没有 AOT 镜像，回放真实跑过的结论';
  } catch (e) { /* health 挂了也让页面能开 */ }
  await refreshCluster();
  setInterval(refreshCluster, 60000);
  S = await api('/api/session', {});
  render();
  $('#input').focus();
})();
