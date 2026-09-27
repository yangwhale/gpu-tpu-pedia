#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""课程视频渲染器：课件网页 ＋ 讲课录音 ＋ 提示表（哪一秒讲到页面哪儿） → mp4。

做法（确定性逐帧）：无头 Chromium 打开课件，关掉页面自己的过渡动画和播放器，
每一帧由时间 t 算出：滚动位置、标注（红框/圈/聚光）、页面里各段动画视频该在第几秒，
设好以后截一帧；相邻帧状态完全相同就复用上一帧。分片并行，最后拼接、合音。

用法：python3 render.py cues.json out.mp4 [--shards 8] [--fps 25]
"""
import json, math, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor

W, H, NAV = 2200, 1238, 96          # 截图视口（最后缩到 1920×1080）；顶部吸顶目录高度
FPS = 25


def ease(x):
    x = max(0.0, min(1.0, x))
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


SETUP_JS = r"""
async (opens) => {
  const st = document.createElement('style');
  st.textContent = `*{transition:none!important;animation:none!important;scroll-behavior:auto!important}
    .lecaudio,.lecmedia{display:none!important}`;
  document.head.appendChild(st);
  for (const sel of opens) document.querySelectorAll(sel).forEach(d => d.open = true);
  // 页面自带的分步播放器会自己暂停/跳转 —— 换成没挂监听的新 video，由我们逐帧设 currentTime
  document.querySelectorAll('video').forEach(v => {
    const n = v.cloneNode(true); n.removeAttribute('autoplay'); n.removeAttribute('loop');
    n.removeAttribute('data-pauses'); n.muted = true; n.preload = 'auto';
    v.replaceWith(n); n.pause();
  });
  document.querySelectorAll('.stepbar, .stepctl').forEach(e => e.style.display = 'none');
  // 右下角浮动按钮一类：视口下半截的 fixed 元素全藏掉
  document.querySelectorAll('body *').forEach(e => {
    const cs = getComputedStyle(e);
    if (cs.position === 'fixed' && e.getBoundingClientRect().top > innerHeight / 2) e.style.display = 'none';
  });
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.id = 'ovl';
  svg.setAttribute('style', 'position:fixed;left:0;top:0;width:100vw;height:100vh;pointer-events:none;z-index:99999');
  document.body.appendChild(svg);
  await Promise.all([...document.querySelectorAll('video')].map(v => new Promise(r => {
    if (v.readyState >= 2) return r(); v.addEventListener('loadeddata', r, {once: true}); v.load(); setTimeout(r, 8000);
  })));
  window.__geom = (sel) => { const e = document.querySelector(sel); if (!e) return null;
    const b = e.getBoundingClientRect(); return [b.left + scrollX, b.top + scrollY, b.width, b.height]; };
  window.__vdur = (sel) => { const v = document.querySelector(sel + ' video') || document.querySelector(sel);
    return v && v.duration ? v.duration : 0; };
  return true;
}
"""

FRAME_JS = r"""
async (s) => {
  window.scrollTo(0, s.y);
  const svg = document.getElementById('ovl'); const ns = 'http://www.w3.org/2000/svg';
  svg.innerHTML = '';
  const W = innerWidth, H = innerHeight;
  if (s.spot) {                       // 聚光：整屏压暗，目标处挖洞
    const [x, y, w, h, a] = s.spot, yy = y - s.y, pad = 14;
    const p = document.createElementNS(ns, 'path');
    p.setAttribute('d', `M0,0H${W}V${H}H0Z M${x-pad},${yy-pad} h${w+2*pad} v${h+2*pad} h${-(w+2*pad)}Z`);
    p.setAttribute('fill', `rgba(20,20,28,${a})`); p.setAttribute('fill-rule', 'evenodd');
    svg.appendChild(p);
  }
  for (const m of s.marks) {
    const [x, y, w, h] = m.r, yy = y - s.y;
    let el;
    if (m.kind === 'circle') {
      el = document.createElementNS(ns, 'ellipse');
      el.setAttribute('cx', x + w / 2); el.setAttribute('cy', yy + h / 2);
      el.setAttribute('rx', w / 2 + 26); el.setAttribute('ry', h / 2 + 18);
      el.setAttribute('transform', `rotate(-2 ${x + w / 2} ${yy + h / 2})`);
    } else if (m.kind === 'under') {
      el = document.createElementNS(ns, 'path');
      el.setAttribute('d', `M${x},${yy + h + 6} Q${x + w / 2},${yy + h + 12} ${x + w},${yy + h + 4}`);
    } else {
      el = document.createElementNS(ns, 'rect');
      el.setAttribute('x', x - 10); el.setAttribute('y', yy - 10);
      el.setAttribute('width', w + 20); el.setAttribute('height', h + 20); el.setAttribute('rx', 14);
    }
    el.setAttribute('fill', 'none'); el.setAttribute('stroke', m.color || '#EA4335');
    el.setAttribute('stroke-width', m.kind === 'under' ? 6 : 5); el.setAttribute('stroke-linecap', 'round');
    svg.appendChild(el);
    const L = el.getTotalLength ? el.getTotalLength() : 4000;
    el.setAttribute('stroke-dasharray', L); el.setAttribute('stroke-dashoffset', L * (1 - m.p));
  }
  const waits = [];
  for (const [sel, t] of s.videos) {
    const v = document.querySelector(sel + ' video') || document.querySelector(sel);
    if (!v) continue;
    if (Math.abs(v.currentTime - t) > 0.001) {
      waits.push(new Promise(r => { v.addEventListener('seeked', r, {once: true}); setTimeout(r, 3000); }));
      v.currentTime = t;
    }
  }
  await Promise.all(waits);
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  return true;
}
"""


def region(geo, cue_target, sub):
    x, y, w, h = geo[cue_target]
    if sub:
        x0, y0, x1, y1 = sub
        return [x + w * x0, y + h * y0, w * (x1 - x0), h * (y1 - y0)]
    return [x, y, w, h]


def plan(cues, geo, vdur):
    """把提示表预处理成每条的：区域、目标滚动位置。"""
    out = []
    prev_y = 0
    for i, c in enumerate(cues):
        r = region(geo, c["target"], c.get("sub"))
        avail = H - NAV
        whole = geo[c["target"]]
        focus = whole if whole[3] + 40 <= avail else r
        if focus[3] + 40 > avail:
            y = focus[1] - NAV - 20
        else:
            y = focus[1] + focus[3] / 2 - (NAV + avail / 2)
        if c.get("scroll") is not None:
            y = geo[c["target"]][1] - NAV - c["scroll"]
        y = max(0, y)
        out.append(dict(c, r=r, y=y, y0=prev_y if i else y))   # 第一条不从页顶滚下来：片头第一帧就停在该讲的位置
        prev_y = y
    return out


def state_at(t, P, vdur):
    i = max(k for k in range(len(P)) if P[k]["t"] <= t) if t >= P[0]["t"] else 0
    c = P[i]
    dt = t - c["t"]
    y = c["y0"] + (c["y"] - c["y0"]) * ease(dt / 0.9)
    marks, spot = [], None
    kind = c.get("kind", "box")
    if kind != "none":
        p = round(min(1.0, max(0.0, (dt - 0.5) / 0.6)), 3)
        marks.append({"r": c["r"], "kind": kind, "p": p, "color": c.get("color")})
    if c.get("spot"):
        a = round(0.45 * min(1.0, max(0.0, dt / 0.6)), 3)
        spot = c["r"] + [a]
    vids = []
    play = None                                    # 往回找最近一条写了 play 的提示（play: null ＝ 停）
    for k in range(i, -1, -1):
        if "play" in P[k]:
            if P[k]["play"]:
                sels = P[k]["play"] if isinstance(P[k]["play"], list) else [P[k]["play"]]
                play = (set(sels), P[k]["t"] + P[k].get("play_delay", 0))
            break
    for sel, d in vdur.items():
        if play and sel in play[0] and t >= play[1] and d > 0:
            vids.append([sel, round(((t - play[1]) % d), 2)])
        else:
            vids.append([sel, 0.0])
    return {"y": round(y), "marks": marks, "spot": spot, "videos": vids}


def shard(args):
    idx, f0, f1, cues_path, html, fps, out = args
    from playwright.sync_api import sync_playwright
    cfg = json.load(open(cues_path))
    cues = cfg["cues"]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg",
                           "-framerate", str(fps), "-i", "-", "-vf", "scale=1920:1080:flags=lanczos",
                           "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", out],
                          stdin=subprocess.PIPE)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        pg = b.new_page(viewport={"width": W, "height": H})
        pg.goto("file://" + html)
        pg.wait_for_timeout(1500)
        pg.evaluate(SETUP_JS, cfg.get("open", []))
        pg.wait_for_timeout(800)
        plays = set()
        for c in cues:
            v = c.get("play")
            if v: plays |= set(v if isinstance(v, list) else [v])
        sels = sorted({c["target"] for c in cues} | plays)
        geo = {s: pg.evaluate("s => window.__geom(s)", s) for s in sels}
        missing = [s for s, g in geo.items() if not g]
        assert not missing, missing
        vdur = {v: pg.evaluate("s => window.__vdur(s)", v) for v in sorted(plays)}
        P = plan(cues, geo, vdur)
        last_key, last_img, n_shot = None, None, 0
        for f in range(f0, f1):
            s = state_at(f / fps, P, vdur)
            key = json.dumps(s, sort_keys=True)
            if key != last_key:
                pg.evaluate(FRAME_JS, s)
                last_img = pg.screenshot(type="jpeg", quality=92)
                last_key = key; n_shot += 1
            ff.stdin.write(last_img)
        b.close()
    ff.stdin.close(); ff.wait()
    return idx, n_shot, f1 - f0


def main():
    cues_path, out = sys.argv[1], sys.argv[2]
    shards = int(sys.argv[sys.argv.index("--shards") + 1]) if "--shards" in sys.argv else 8
    fps = int(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else FPS
    only = float(sys.argv[sys.argv.index("--until") + 1]) if "--until" in sys.argv else None
    cfg = json.load(open(cues_path))
    html, audio = os.path.abspath(cfg["html"]), os.path.abspath(cfg["audio"])
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", audio],
                               capture_output=True, text=True).stdout)
    if only:
        dur = min(dur, only)
    N = int(math.ceil(dur * fps))
    step = int(math.ceil(N / shards))
    tmp = os.path.splitext(out)[0] + "-parts"
    os.makedirs(tmp, exist_ok=True)
    jobs = [(i, i * step, min(N, (i + 1) * step), cues_path, html, fps, os.path.join(tmp, "p%02d.mp4" % i))
            for i in range(shards) if i * step < N]
    t0 = time.time()
    with ProcessPoolExecutor(len(jobs)) as ex:
        for idx, n, tot in ex.map(shard, jobs):
            print("shard %d: %d 张截图 / %d 帧" % (idx, n, tot), flush=True)
    lst = os.path.join(tmp, "list.txt")
    open(lst, "w").write("".join("file '%s'\n" % j[-1] for j in jobs))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-i", audio, "-t", str(dur), "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", out], check=True)
    print("ok %s  %.0f 秒视频  用时 %.0f 秒" % (out, dur, time.time() - t0))


if __name__ == "__main__":
    main()
