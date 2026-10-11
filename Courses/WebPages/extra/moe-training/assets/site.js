// v7x-gsc 网页交互：代码复制、阅读进度、本页目录高亮、全站搜索（懒加载 search-index.js）
(function () {
  var root = document.body.getAttribute('data-root') || '.';

  // 复制按钮
  document.querySelectorAll('.code').forEach(function (box) {
    var pre = box.querySelector('pre'); if (!pre) return;
    var b = document.createElement('button'); b.className = 'copy'; b.type = 'button'; b.textContent = '复制';
    b.addEventListener('click', function () {
      var txt = pre.innerText;
      var done = function () { b.textContent = '已复制'; b.classList.add('done');
        setTimeout(function () { b.textContent = '复制'; b.classList.remove('done'); }, 1400); };
      if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(txt).then(done);
      else { var t = document.createElement('textarea'); t.value = txt; document.body.appendChild(t); t.select();
        try { document.execCommand('copy'); done(); } catch (e) {} document.body.removeChild(t); }
    });
    box.appendChild(b);
  });

  // R17：窄屏上顶栏横向滚动，当前栏目可能被挤到搜索框下面 → 加载时把当前 tab 滚到顶栏中间（只滚顶栏，不动页面）
  var secs = document.querySelector('nav.secs'), secOn = secs && secs.querySelector('a.on');
  if (secOn && secs.scrollWidth > secs.clientWidth) {
    var nr = secs.getBoundingClientRect(), ar = secOn.getBoundingClientRect();
    secs.scrollLeft += (ar.left - nr.left) - (nr.width - ar.width) / 2;
  }

  // 进度条 + 回顶 + 目录高亮
  var bar = document.querySelector('.progress i'), top = document.querySelector('.totop');
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll('nav.ptoc a'));
  var heads = tocLinks.map(function (a) { return document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1))); });
  function onScroll() {
    var h = document.documentElement, y = h.scrollTop, max = h.scrollHeight - h.clientHeight;
    if (bar) bar.style.width = (max > 0 ? (y / max * 100) : 0) + '%';
    if (top) top.classList.toggle('on', y > 600);
    var cur = -1;
    for (var i = 0; i < heads.length; i++) { if (heads[i] && heads[i].getBoundingClientRect().top < 120) cur = i; }
    tocLinks.forEach(function (a, i) { a.classList.toggle('on', i === cur); });
  }
  window.addEventListener('scroll', onScroll, { passive: true }); onScroll();
  // 当前页在侧栏里居中：挪到文件末尾的侧栏模块里做（侧栏的滚动容器换成了 .side-in）

  // 搜索
  var q = document.getElementById('q'), qr = document.getElementById('qr'), loaded = false, sel = -1;
  function load(cb) {
    if (loaded) return cb();
    var s = document.createElement('script'); s.src = root + '/assets/search-index.js';
    s.onload = function () { loaded = true; cb(); }; document.head.appendChild(s);
  }
  function esc(s) { return s.replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function mark(s, terms) {
    var out = esc(s);
    terms.forEach(function (t) { if (!t) return;
      var re = new RegExp(t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
      out = out.replace(re, function (m) { return '<mark>' + m + '</mark>'; }); });
    return out;
  }
  // 第 16 轮：报错原文匹配。粘一整行日志时 ID / 时间戳 / 数字每次都不一样，不能要求逐字相同：
  // 把查询切成词（丢掉纯数字、十六进制串、UUID 这类每次都变的），看报错条目覆盖了查询词的几成。
  function toks(s) {
    return (s.toLowerCase().match(/[a-z_][a-z0-9_]*/g) || []).filter(function (t) {
      return t.length >= 3 && !/^[0-9a-f]{8,}$/.test(t) && !/^(the|and|for|with|from|this|that|was|are|not|but)$/.test(t);
    });
  }
  function uniq(a) { var o = {}, r = []; a.forEach(function (t) { if (!o[t]) { o[t] = 1; r.push(t); } }); return r; }
  function errMatch(v) {
    var q = uniq(toks(v)); if (q.length < 2) return [];
    var norm = v.toLowerCase().replace(/\s+/g, ' ').trim(), res = [], best = {};
    (window.__SEARCH__ || []).forEach(function (d) {
      if (d.k !== 'e') return;
      if (!d._t) d._t = uniq(toks(d.x));
      var set = {}; d._t.forEach(function (t) { set[t] = 1; });
      var m = q.filter(function (t) { return set[t]; }).length;
      var cov = m / q.length, prec = m / Math.max(1, d._t.length);
      var exact = norm.length >= 12 && d.x.toLowerCase().replace(/\s+/g, ' ').indexOf(norm) >= 0;
      if (!exact && (m < 2 || (cov < 0.5 && m < 4))) return;
      var sc = (exact ? 60 : 0) + 40 * cov + 15 * prec + m;
      var key = d.u;                                // 同一问题页只留最像的一条
      if (!best[key] || best[key].s < sc) best[key] = { d: d, s: sc, a: '', snip: d.x.slice(0, 160) + (d.x.length > 160 ? '…' : ''), err: 1, terms: q };
    });
    Object.keys(best).forEach(function (k) { res.push(best[k]); });
    return res.sort(function (a, b) { return b.s - a.s; }).slice(0, 8);
  }
  function search(v) {
    var terms = v.toLowerCase().split(/\s+/).filter(Boolean); if (!terms.length) return [];
    var res = [];
    (window.__SEARCH__ || []).forEach(function (d) {
      if (d.k === 'e') return;
      var tl = d.t.toLowerCase(), xl = d.x.toLowerCase(), score = 0, ok = true, anchor = '', firstPos = -1;
      terms.forEach(function (t) {
        var inT = tl.indexOf(t) >= 0, p = xl.indexOf(t);
        if (!inT && p < 0) { ok = false; return; }
        if (inT) score += 10;
        if (p >= 0) { score += 1 + Math.min(5, xl.split(t).length - 1) * 0.4; if (firstPos < 0) firstPos = p; }
      });
      if (!ok) return;
      if (d.k === 'c') score += 4;                   // 概念条目：术语名命中时排前
      if (/^gsc-\d+$/.test(terms[0]) && tl.indexOf(terms[0]) === 0) score += 50;   // 直接输问题号
      for (var i = 0; i < d.h.length; i++) { if (d.h[i][1].toLowerCase().indexOf(terms[0]) >= 0) { anchor = '#' + d.h[i][0]; score += 3; break; } }
      var snip = '';
      if (firstPos >= 0) { var a = Math.max(0, firstPos - 40); snip = (a > 0 ? '…' : '') + d.x.slice(a, a + 120) + '…'; }
      res.push({ d: d, s: score, a: anchor, snip: snip, terms: terms });
    });
    res.sort(function (a, b) { return b.s - a.s; });
    var errs = errMatch(v), seen = {};
    // 报错命中且覆盖了查询的大半 → 排最前；其余按普通结果排
    var strong = errs.filter(function (r) { return r.s >= 45; }), weak = errs.filter(function (r) { return r.s < 45; });
    var out = strong.concat(res.slice(0, 20), weak).filter(function (r) { var k = r.d.u + r.a + (r.err ? 'e' : ''); if (seen[k]) return false; seen[k] = 1; return true; });
    return out.slice(0, 20);
  }
  window.__siteSearch = search;   // 给 site/shots.py 的搜索用例用
  function show() {
    var v = q.value.trim(); sel = -1;
    if (!v) { qr.classList.remove('on'); return; }
    load(function () {
      var rs = search(v);
      qr.innerHTML = rs.length ? rs.map(function (r) {
        return '<a href="' + root + '/' + r.d.u + r.a + '"' + (r.err ? ' class="err"' : '') + '><b><span class="s">' +
          (r.err ? '报错原文 →' : esc(r.d.s)) + '</span>' + mark(r.d.t, r.err ? [] : r.terms) +
          '</b><span class="x">' + mark(r.snip, r.terms) + '</span></a>';
      }).join('') : '<div class="empty">没有结果（多个词用空格分开，全部命中才算）</div>';
      qr.classList.add('on');
    });
  }
  if (q) {
    q.addEventListener('focus', function () { load(function () {}); });
    q.addEventListener('input', show);
    q.addEventListener('keydown', function (e) {
      var items = qr.querySelectorAll('a');
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault(); if (!items.length) return;
        sel = (sel + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
        items.forEach(function (a, i) { a.classList.toggle('sel', i === sel); });
        items[sel].scrollIntoView({ block: 'nearest' });
      } else if (e.key === 'Enter') { var t = items[sel >= 0 ? sel : 0]; if (t) location.href = t.href; }
      else if (e.key === 'Escape') { qr.classList.remove('on'); q.blur(); }
    });
    document.addEventListener('click', function (e) { if (!e.target.closest('.search')) qr.classList.remove('on'); });
    document.addEventListener('keydown', function (e) {
      if (e.key === '/' && document.activeElement !== q && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) { e.preventDefault(); q.focus(); }
    });
  }
})();

// 第 15 轮：数据图悬停提示（把 <title> 挪成自定义浮层，免得原生提示慢半拍还和浮层重影；读屏改用 aria-label）
(function () {
  var pts = document.querySelectorAll('svg.chart .pt');
  if (!pts.length) return;
  var tip = document.createElement('div'); tip.className = 'ctip'; tip.setAttribute('role', 'status'); document.body.appendChild(tip);
  function place(x, y) {
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var nx = x + 14, ny = y - h - 10;
    if (nx + w > innerWidth - 8) nx = x - w - 14;
    if (ny < 60) ny = y + 18;
    tip.style.left = nx + 'px'; tip.style.top = ny + 'px';
  }
  addEventListener('scroll', function () { tip.classList.remove('on'); }, { passive: true });
  pts.forEach(function (g) {
    var t = g.querySelector('title'); if (!t) return;
    var s = t.textContent; g.setAttribute('aria-label', s); g.removeChild(t);
    g.addEventListener('pointerenter', function (e) { tip.textContent = s; tip.classList.add('on'); place(e.clientX, e.clientY); });
    g.addEventListener('pointermove', function (e) { place(e.clientX, e.clientY); });
    g.addEventListener('pointerleave', function () { tip.classList.remove('on'); });
    g.addEventListener('focus', function () { var r = g.getBoundingClientRect(); tip.textContent = s; tip.classList.add('on'); place(r.right, r.top); });
    g.addEventListener('blur', function () { tip.classList.remove('on'); });
  });
})();

// 第 16 轮：阅读路径条。从首页三条路径点进来时 URL 带 ?path=first&i=3，页面顶上显示第几站 / 上一站 / 下一站。
(function () {
  var m = /[?&]path=(\w+)&i=(\d+)/.exec(location.search); if (!m) return;
  var root = document.body.getAttribute('data-root') || '.';
  var s = document.createElement('script'); s.src = root + '/assets/paths.js';
  s.onload = function () {
    var P = (window.__PATHS__ || {})[m[1]]; if (!P) return; var i = +m[2], it = P.items;
    function href(k) { var u = it[k].u, h = u.indexOf('#'); var p = h < 0 ? u : u.slice(0, h), f = h < 0 ? '' : u.slice(h);
      return root + '/' + p + '?path=' + m[1] + '&i=' + k + f; }
    function esc(x) { return x.replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
    var bar = document.createElement('nav'); bar.className = 'pathbar'; bar.setAttribute('aria-label', '阅读路径');
    var dots = it.map(function (x, k) { return '<a class="pdot' + (k === i ? ' on' : '') + '" href="' + href(k) + '" title="' + esc((k + 1) + '. ' + x.t) + '"></a>'; }).join('');
    bar.innerHTML = '<span class="pn0"><a href="' + root + '/index.html#paths">阅读路径</a> · <b>' + esc(P.name) + '</b> 第 ' + (i + 1) + ' / ' + it.length + ' 站：' + esc(it[i].t) + '</span>' +
      '<span class="pdots">' + dots + '</span>' +
      (i > 0 ? '<a class="pp" href="' + href(i - 1) + '">← ' + esc(it[i - 1].t) + '</a>' : '') +
      (i + 1 < it.length ? '<a class="pnx" href="' + href(i + 1) + '">下一站：' + esc(it[i + 1].t) + ' →</a>' : '<span class="pend">路径终点</span>');
    var main = document.getElementById('main'); if (main) main.insertBefore(bar, main.firstChild);
    document.documentElement.classList.add('has-path');
    // 路径条插进来会把正文往下推约 50 px：带锚点进来的，重新对准一次（不然目标标题被路径条压住）
    if (location.hash) { var t = document.getElementById(decodeURIComponent(location.hash.slice(1)));
      if (t) { document.documentElement.style.scrollBehavior = 'auto'; t.scrollIntoView(); document.documentElement.style.scrollBehavior = ''; } }
  };
  document.head.appendChild(s);
})();

// 第 17 轮：讲义页「⇆ 并排看手册」—— 左讲义、右手册；点 🖥 屏幕里的链接，右边手册跟着跳。状态记在 localStorage。
(function () {
  if (window.self !== window.top) { document.documentElement.classList.add('embedded'); return; }
  var btn = document.querySelector('.lsplit-btn'), pane = document.querySelector('aside.lsplit');
  if (!btn || !pane) return;
  var fr = pane.querySelector('iframe'), KEY = 'v7x-lsplit';
  function set(on) {
    document.body.classList.toggle('lsplit-on', on);
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    btn.textContent = on ? '✕ 关闭并排' : '⇆ 并排看手册';
    if (on && !fr.getAttribute('src')) fr.setAttribute('src', fr.getAttribute('data-src'));
    try { localStorage.setItem(KEY, on ? '1' : '0'); } catch (e) {}
  }
  btn.addEventListener('click', function () { set(!document.body.classList.contains('lsplit-on')); });
  document.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('a[data-chap]');
    if (!a || !document.body.classList.contains('lsplit-on') || e.metaKey || e.ctrlKey) return;
    e.preventDefault(); fr.setAttribute('src', a.getAttribute('href'));
    document.querySelectorAll('p.board.cur').forEach(function (x) { x.classList.remove('cur'); });
    var p = a.closest('p.board'); if (p) p.classList.add('cur');
  });
  var saved = null; try { saved = localStorage.getItem(KEY); } catch (e) {}
  if (saved === '1' && window.innerWidth >= 1100) set(true);
})();

// 第八个月：侧栏收起 / 拖宽 / 窄屏抽屉。宽度存 v7x-sidew（px），收起存 v7x-side=off；head 里的内联脚本先套上，避免闪一下
(function () {
  var side = document.querySelector('nav.side'); if (!side) return;
  var d = document.documentElement, MIN = 160, MAX = 560;
  function save(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch (e) {} }
  var inner = document.createElement('div'); inner.className = 'side-in';
  while (side.firstChild) inner.appendChild(side.firstChild);
  side.appendChild(inner); side.classList.add('js-side');
  var cur = side.querySelector('a.on');      // 当前页滚到侧栏中间（只滚侧栏，不动页面）
  if (cur) inner.scrollTop = cur.offsetTop - inner.clientHeight / 2;
  var narrow = window.matchMedia('(max-width:900px)');
  var x = document.createElement('button'); x.type = 'button'; x.className = 'side-x'; x.textContent = '«';
  x.title = '收起侧栏（快捷键 [ ）'; x.setAttribute('aria-label', '收起侧栏');
  var grip = document.createElement('div'); grip.className = 'side-grip'; grip.title = '拖动调宽，双击复原';
  side.appendChild(x); side.appendChild(grip);
  var open = document.createElement('button'); open.type = 'button'; open.className = 'side-open'; open.textContent = '☰ 目录';
  open.title = '展开侧栏（快捷键 [ ）'; document.body.appendChild(open);
  function setOff(off) { d.classList.toggle('side-off', off); save('v7x-side', off ? 'off' : null); }
  function toggle() {
    if (narrow.matches) d.classList.toggle('side-drawer');
    else setOff(!d.classList.contains('side-off'));
  }
  x.addEventListener('click', function () { setOff(true); });
  open.addEventListener('click', function (e) { e.stopPropagation(); if (narrow.matches) d.classList.toggle('side-drawer'); else setOff(false); });
  document.addEventListener('click', function (e) {
    if (d.classList.contains('side-drawer') && !side.contains(e.target)) d.classList.remove('side-drawer');
  });
  document.addEventListener('keydown', function (e) {
    var t = e.target; if (e.key !== '[' || e.ctrlKey || e.metaKey || e.altKey) return;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    e.preventDefault(); toggle();
  });
  grip.addEventListener('dblclick', function () { d.style.removeProperty('--sidew'); save('v7x-sidew', null); });
  grip.addEventListener('pointerdown', function (e) {
    e.preventDefault(); grip.setPointerCapture(e.pointerId); d.classList.add('side-drag');
    var x0 = e.clientX, w0 = side.getBoundingClientRect().width, w = w0;
    function mv(ev) {
      w = w0 + ev.clientX - x0;
      if (w < MIN - 60) { d.classList.add('side-off'); return; }      // 往左拖到底 = 收起（像 VS Code）
      d.classList.remove('side-off');
      d.style.setProperty('--sidew', Math.max(MIN, Math.min(MAX, Math.round(w))) + 'px');
    }
    function up() {
      grip.removeEventListener('pointermove', mv); grip.removeEventListener('pointerup', up); grip.removeEventListener('pointercancel', up);
      d.classList.remove('side-drag');
      if (d.classList.contains('side-off')) { setOff(true); return; }
      save('v7x-sidew', String(Math.max(MIN, Math.min(MAX, Math.round(w)))));
    }
    grip.addEventListener('pointermove', mv); grip.addEventListener('pointerup', up); grip.addEventListener('pointercancel', up);
  });
})();
