# -*- coding: utf-8 -*-
import io
GiB=2**30; HBM=206_000_000_000
# ⛔ 这个 206×10⁹ **不是官方口径**，是 JAX 公开源码里那一行常量。
#    官方两处写的都是 192（规格表 GiB、正文 GB），而 192 GiB = 206.158×10⁹ B ——
#    源码那个 206.000×10⁹ 是**取整过的**。
#    后果要说清：由它反推出来的「运行时预留」= 95.93 − 94.74 = 1.19 GiB；
#    换成官方的 192 GiB 起算就是 96.00 − 94.74 = 1.26 GiB。
#    也就是说这一步的余量本身带着约 0.07 GiB（6%）的口径噪声，
#    **不能拿它当一个精确常数去做规划**，只能用来说明「有这么一层，别忘了」。
CHIP=HBM/GiB; DEV=CHIP/2; ALLOC=94.74; RES=DEV-ALLOC
X0,BARW=196,572; ROWH,Y0=78,66
def w(g): return BARW*(g/CHIP)
rows=[("JAX 源码口径","206 × 10⁹ B（官方两处都写 192）","206 × 10⁹ B",CHIP,"#1a73e8",None,None,None),
      ("换成二进制","191.85 ≈ 192 GiB/chip",f"{CHIP:.2f} GiB",CHIP,"#1a73e8",None,
       "当成 206 GiB 用","凭空多 +7.4%"),
      ("÷ 2 个 device",f"{DEV:.2f} GiB/device · 物理",f"{DEV:.2f} GiB",DEV,"#9334e6",CHIP,
       "忘了 2 device / chip","凭空多 +100%"),
      ("− 运行时预留",f"{ALLOC} GiB/device · 可分配",f"{ALLOC} GiB",ALLOC,"#1e8e3e",DEV,
       f"忘了预留 ~{RES:.2f} GiB","凭空多 +1.3%")]
assert abs(RES-1.19)<0.01, "预留算出来不是 1.19 了，先确认换的是哪个 192"
H=Y0+ROWH*len(rows)+104
p=[f'<svg viewBox="0 0 1000 {H}" width="100%" role="img" aria-label="94.74 GiB 的三步推导：十进制换二进制、除以每颗芯片的两个 device、减运行时预留">',
   '<defs><marker id="a0" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#80868b"/></marker></defs>',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">同一块 HBM，四种说法 —— 全都对，但只有最后一个能拿去判 OOM</text>',
   '<text class="svgsm" x="0" y="36">条的长度 = 真实字节数，等比作图</text>',
   f'<text class="svgsm" x="{X0+BARW+20}" y="36" fill="#d93025">这一步搞错的代价</text>']
for i,(t,sub,inbar,g,col,ghost,err,es) in enumerate(rows):
    y=Y0+i*ROWH
    if ghost:
        p.append(f'<rect x="{X0+w(g):.1f}" y="{y}" width="{w(ghost)-w(g):.1f}" height="36" rx="4" '
                 f'fill="#fff" stroke="#dadce0" stroke-dasharray="3 3"/>')
    p.append(f'<rect x="{X0}" y="{y}" width="{w(g):.1f}" height="36" rx="4" fill="{col}"/>')
    p.append(f'<text class="svglbl" x="{X0-16}" y="{y+15}" text-anchor="end" fill="#202124">{t}</text>')
    p.append(f'<text class="svgsm" x="{X0-16}" y="{y+31}" text-anchor="end">{sub}</text>')
    p.append(f'<text class="svgnum" x="{X0+14}" y="{y+24}" fill="#fff">{inbar}</text>')
    if err:
        p.append(f'<text class="svglbl" x="{X0+BARW+20}" y="{y+16}" fill="#d93025">{err}</text>')
        p.append(f'<text class="svgsm" x="{X0+BARW+20}" y="{y+32}" fill="#d93025">{es}</text>')
    if i: p.append(f'<path d="M{X0-104} {y-24} v16" stroke="#80868b" stroke-width="1.2" marker-end="url(#a0)"/>')
# 第 2 行：进制不改变长度
y2=Y0+ROWH
p.append(f'<path d="M{X0+BARW} {Y0+36} v{ROWH-36} " stroke="#f9ab00" stroke-width="1.4" stroke-dasharray="4 3"/>')
p.append(f'<rect x="{X0+BARW-330}" y="{y2-24}" width="330" height="18" rx="3" fill="#fef7e0" stroke="#f9ab00"/>')
p.append(f'<text class="svgsm" x="{X0+BARW-16}" y="{y2-11}" text-anchor="end" fill="#7a5000">'
         f'⚠️ 一个字节都没少，只是换了进制 —— 两条一样长</text>')
# 第 4 行：那条几乎看不见的预留
y4=Y0+3*ROWH
xa,xb=X0+w(ALLOC),X0+w(DEV)
p.append(f'<path d="M{xa:.1f} {y4+42} v10 H{xb:.1f} v-10" fill="none" stroke="#d93025" stroke-width="1.3"/>')
p.append(f'<path d="M{(xa+xb)/2:.1f} {y4+52} L{(xa+xb)/2+96:.1f} {y4+64}" stroke="#d93025" stroke-width="1.2"/>')
# 这三行必须拆开：起点 x≈660，右边只剩 340px，一行写不下会直接冲出画布
# （渲染出来才看得见 —— 所以改完这里一定要重截图）。
for _k,_t in enumerate([f'⚠️ 这条窄到几乎看不见的缝，就是 {RES:.2f} GiB',
                        '从官方那个 192 GiB 起算则是 1.26 —— 带 6% 口径噪声',
                        '最容易忘的一步，往往也是最不起眼的那一步']):
    p.append(f'<text class="svgsm" x="{(xa+xb)/2+102:.1f}" y="{y4+62+_k*15}" '
             f'fill="#d93025">{_t}</text>')
yb=Y0+ROWH*len(rows)+28
p.append(f'<line x1="0" y1="{yb}" x2="1000" y2="{yb}" stroke="#e8eaed"/>')
p.append(f'<text class="svglbl" x="0" y="{yb+26}" fill="#1e8e3e">落笔前的三个自问　'
         f'① 二进制还是十进制？　② per chip / per device / per core？　③ 物理量还是可用量？</text>')
p.append(f'<text class="svgsm" x="0" y="{yb+46}">这三问不是凑数 —— 上面三步各对应一个。'
         f'第 4 节讲 NVL72 时会在集群这一层再撞一次同样的病'
         f'（72 张是物理域；我们这批集群按 64 编排 —— 那是运维选择，不是平台限制）</text>')
p.append('</svg>')
io.open('fig0-1.svg','w',encoding='utf-8').write('\n'.join(p)); print('ok',H)
