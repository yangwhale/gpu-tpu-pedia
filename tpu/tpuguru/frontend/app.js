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
let busy = false;
let hideVoided = false;

/* ── 参数控件表（PARAMS.md 的子集，问号文案三段式）───────── */
const FIELDS = [
  { k: '__topology', label: '目标拓扑', type: 'topo', req: true,
    q: '决定 AOT 按多少张卡编译。名字里的数字是 device，v7 是 2 device/chip —— tpu7x-128 是 64 芯片，不是 128。选错了整份显存结论都不成立。' },
  { k: 'model_name', label: '模型', type: 'text',
    q: 'MaxText 里的模型名，决定层数、专家数、hidden 等形状。lint 的整除类规则要靠它拿到专家数。' },
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
  const r = await fetch(path, opt);
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
  inp.onchange = async () => {
    setBusy(true);
    try { S = await api('/api/set', { session_id: S.session_id, param: f.k, value: inp.value }); render(); }
    catch (e) { toast('改失败：' + e.message); } finally { setBusy(false); }
  };
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
    p.appendChild(el('div', 'empty',
      '<span class="ic">📊</span>还没跑过 AOT<br>右上角【跑 AOT】，大约 3 分钟，不占卡'));
    return;
  }

  if (R.fingerprint && S.fingerprint && R.fingerprint !== S.fingerprint) {
    p.appendChild(el('div', 'stale',
      '<div>⚠️</div><div><b>这份报告不是当前配置跑出来的。</b><br>'
      + '跑完之后配置又被改过了。要看当前这套的结论，重新点【跑 AOT】。<br>'
      + `<span style="font-family:var(--mono);font-size:11.5px;color:var(--muted)">`
      + `报告 ${esc(R.fingerprint)} · 当前 ${esc(S.fingerprint)}</span></div>`));
  }
  // 结论
  const v = el('div', 'card');
  const ok = R.ok;
  const txt = ok === true ? '✅ 编译通过，装得下' : ok === false ? '❌ 装不下' : '❓ 这一档没有实测记录';
  v.appendChild(el('div', 'big ' + (ok === true ? 'ok' : ok === false ? 'bad' : 'unk'), txt));
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
    v.appendChild(k);
  }
  const badge = el('div', 'hint');
  badge.style.marginTop = '12px';
  badge.innerHTML = R.mode === 'replay'
    ? `<span class="pill warn"><span class="dot"></span>replay 模式</span> 本机没有 AOT 镜像，回放的是真实跑过的结论。`
      + (R.source ? `<br>出处：${esc(R.source)}` : '')
    : `<span class="pill"><span class="dot"></span>real 模式</span> 本机 docker 真跑的。`;
  v.appendChild(badge);
  p.appendChild(v);

  const A = R.analyses || {};
  if (A.scale) p.appendChild(cardScale(A.scale.data));
  if (A.memory) p.appendChild(cardMemory(A.memory.data));
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
  // 按 parent_save_id 摆成树
  const byId = Object.fromEntries(list.map(s => [s.id, s]));
  const kids = {}; const roots = [];
  list.forEach(s => {
    if (s.parent_save_id && byId[s.parent_save_id]) (kids[s.parent_save_id] ||= []).push(s);
    else roots.push(s);
  });
  const tree = el('div', 'tree');
  const walk = (s, depth) => {
    tree.appendChild(nodeOf(s, depth));
    (kids[s.id] || []).sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''))
      .forEach(k => walk(k, Math.min(depth + 1, 3)));
  };
  roots.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || '')).forEach(r => walk(r, 0));
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
  if (rs.ok === true) bits.push('✅'); else if (rs.ok === false) bits.push('❌');
  left.appendChild(el('div', 'meta', esc(bits.join('  ·  '))));
  if (s.note) left.appendChild(el('div', 'meta', esc(s.note)));
  if (s.tags && s.tags.length) left.appendChild(el('div', 'meta',
    s.tags.map(t => `<span class="tag ok" style="margin-right:5px">${esc(t)}</span>`).join('')));
  if (s.voided) left.appendChild(el('div', 'meta', '作废原因：' + esc(s.voided.reason || '')));
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
      const r = prompt('作废原因（会一起存下来）：');
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
    const stale = S.result.fingerprint && S.fingerprint && S.result.fingerprint !== S.fingerprint;
    br.hidden = false;
    br.textContent = stale ? '?' : S.result.ok === false ? '!' : '✓';
    br.className = 'badge ' + (stale ? 'amber' : S.result.ok === false ? 'red'
      : S.result.ok ? 'green' : '');
    $('.tab[data-pane="rep"]').title = stale ? '配置改过了，这份报告已经不对应当前配置' : '';
  } else { br.hidden = true; br.textContent = ''; }
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

/* ── 启动 ─────────────────────────────────────────────────── */
(async () => {
  try {
    const h = await api('/api/health');
    TOPOS = h.topologies || {};
    const ps = $('#pill-store');
    ps.className = 'pill' + (h.store === 'firestore' ? '' : ' warn');
    ps.lastElementChild.textContent = h.store === 'firestore' ? 'Firestore' : '本地存储';
    const pm = $('#pill-mode');
    pm.className = 'pill' + (h.aot_mode === 'real' ? '' : ' warn');
    pm.lastElementChild.textContent = h.aot_mode === 'real' ? 'AOT: real' : 'AOT: replay';
    pm.title = h.aot_mode === 'real' ? '本机 docker 真跑' : '没有 AOT 镜像，回放真实跑过的结论';
  } catch (e) { /* health 挂了也让页面能开 */ }
  S = await api('/api/session', {});
  render();
  $('#input').focus();
})();
