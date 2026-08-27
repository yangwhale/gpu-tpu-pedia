import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams.update({'font.family':'Noto Sans CJK SC','font.size':11,
    'axes.grid':True,'grid.alpha':.25,'grid.linestyle':'-',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#9AA0A6','axes.labelcolor':'#202124','text.color':'#202124',
    'xtick.color':'#5F6368','ytick.color':'#5F6368','figure.facecolor':'white'})
BLUE,RED,YELLOW,GREEN,GREY='#1A73E8','#D93025','#F9AB00','#1E8E3E','#5F6368'
E=json.load(open('EXTRACT.json'))
C=E['cases']
K=lambda x,p: f'{x/1000:,.0f}K' if x>=1000 else f'{x:,.0f}'
LEN={8192:'8K',16384:'16K',32768:'32K',65536:'64K',131072:'128K',258048:'252K'}
L2=lambda v: LEN.get(v, f'{v:,}')

# ---- 图1 prefill 吞吐 vs 并发 ----
fig,ax=plt.subplots(figsize=(7.2,4.4))
for key,lab,col,mk,dy,va in [('2_dp8_prefill_8k_sweep','DP8  (DP=8 / PCP=1)',BLUE,'o',14,'bottom'),
                       ('5_pcp8_prefill_8k_sweep','PCP8 (DP=1 / PCP=8)',RED,'s',-34,'bottom')]:
    r=C[key]['rows']; x=[i['concurrency'] for i in r]; y=[i['total_token_throughput'] for i in r]
    ax.plot(x,y,marker=mk,color=col,lw=2.2,ms=7,label=lab)
    pk=C[key]['peak']; ax.annotate(f"峰值 {pk['tok_s']:,.0f} @C{pk['concurrency']}",
        (pk['concurrency'],pk['tok_s']),textcoords='offset points',xytext=(10,dy),
        fontsize=9.5,color=col,ha='left',va=va,fontweight='bold')
ax.set_xscale('log',base=2); ax.set_xticks([8,16,32,64,128,256]); ax.set_xticklabels([8,16,32,64,128,256])
ax.yaxis.set_major_formatter(FuncFormatter(K))
ax.set_xlabel('并发'); ax.set_ylabel('总 token 吞吐 (tok/s, 4 chips)')
ax.set_title('Prefill 吞吐 vs 并发  ·  ISL 8192 / OSL 1  ·  Qwen3.5-397B-A17B-FP8',
             fontsize=12,pad=12,loc='left')
ax.legend(frameon=False,loc='lower right'); ax.set_ylim(0,68000)
fig.tight_layout(); fig.savefig('c1_prefill_throughput.png',dpi=160); plt.close(fig)

# ---- 图2 单请求 TTFT vs 输入长度 ----
fig,ax=plt.subplots(figsize=(7.2,4.4))
for key,lab,col,mk in [('3_dp8_single_request_ttft','DP8',BLUE,'o'),
                       ('6_pcp8_single_request_ttft','PCP8',RED,'s')]:
    r=sorted(C[key]['results'],key=lambda i:i['input_length'])
    ax.plot([i['input_length'] for i in r],[i['median_ttft_ms']/1000 for i in r],
            marker=mk,color=col,lw=2.2,ms=7,label=lab)
d={i['input_length']:i['median_ttft_ms'] for i in C['3_dp8_single_request_ttft']['results']}
p={i['input_length']:i['median_ttft_ms'] for i in C['6_pcp8_single_request_ttft']['results']}
for L in sorted(d):
    ax.annotate(f'{d[L]/p[L]:.1f}×',(L,(d[L]*p[L])**.5/1000),ha='center',va='center',
                fontsize=9,color=GREEN,fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.18',fc='white',ec='none',alpha=.85))
ax.set_xscale('log',base=2); ax.set_yscale('log')
ax.set_xticks(sorted(d)); ax.set_xticklabels([L2(l) for l in sorted(d)])
ax.set_yticks([0.2,0.5,1,2,5,10,20,50]); ax.set_yticklabels(['0.2','0.5','1','2','5','10','20','50'])
ax.set_xlabel('输入长度 (token)'); ax.set_ylabel('median TTFT (秒, 对数轴)')
ax.set_title('单请求 TTFT vs 输入长度  ·  并发 1 / OSL 1  ·  绿色为 PCP8 领先倍数',
             fontsize=12,pad=12,loc='left')
ax.legend(frameon=False,loc='upper left')
fig.tight_layout(); fig.savefig('c2_ttft.png',dpi=160); plt.close(fig)

# ---- 图3 SPEED-Bench ----
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))
cs=[8,64]; w=.36
dp=[r['input_tok_s'] for r in C['4_dp8_speed_bench_mix']['rows']]
pc=[r['input_tok_s'] for r in C['7_pcp8_speed_bench_mix']['rows']]
xs=range(len(cs))
a1.bar([i-w/2 for i in xs],dp,w,color=BLUE,label='DP8')
a1.bar([i+w/2 for i in xs],pc,w,color=RED,label='PCP8')
for i,(a,b) in enumerate(zip(dp,pc)):
    a1.text(i-w/2,a+700,f'{a:,.0f}',ha='center',fontsize=9,color=BLUE)
    a1.text(i+w/2,b+700,f'{b:,.0f}',ha='center',fontsize=9,color=RED)
a1.set_xticks(list(xs)); a1.set_xticklabels([f'并发 {c}' for c in cs])
a1.yaxis.set_major_formatter(FuncFormatter(K)); a1.set_ylabel('input tok/s'); a1.set_ylim(0,58000)
a1.set_title('SPEED-Bench 输入吞吐  ·  C8 时 PCP8 领先，C64 反转',fontsize=11,loc='left',pad=10)
a1.legend(frameon=False,loc='lower right')
lbl=['P50','P90','P99']; w2=.2
for j,(rows,col,nm) in enumerate([(C['4_dp8_speed_bench_mix']['rows'],BLUE,'DP8'),
                                  (C['7_pcp8_speed_bench_mix']['rows'],RED,'PCP8')]):
    for k,cc in enumerate(cs):
        v=[rows[k][f'p{q}_ttft_ms']/1000 for q in [50,90,99]]
        off=(j*2+k-1.5)*w2
        a2.bar([i+off for i in range(3)],v,w2,color=col,alpha=1-0.45*k,
               label=f'{nm} C{cc}')
a2.set_xticks(range(3)); a2.set_xticklabels(lbl)
a2.set_ylabel('TTFT (秒)')
a2.set_title('SPEED-Bench TTFT 分位',fontsize=11,loc='left',pad=10)
a2.legend(frameon=False,fontsize=9,ncol=2)
fig.tight_layout(); fig.savefig('c3_speedbench.png',dpi=160); plt.close(fig)

# ---- 图4 复现度 ----
items=[('DP decode 吞吐',5162.4,5162.40),('DP decode TPOT',41.836,41.76),
       ('DP prefill 峰值',57735.8,57682.03),('PCP prefill 峰值',54350.3,54330.21)]
dd={i['input_length']:i['median_ttft_ms'] for i in C['3_dp8_single_request_ttft']['results']}
pp={i['input_length']:i['median_ttft_ms'] for i in C['6_pcp8_single_request_ttft']['results']}
bd={8192:1004.77,16384:2073.45,32768:4402.55,65536:9910.08,131072:24144.02,258048:64042.64}
bp={8192:217.92,16384:408.26,32768:802.85,65536:1722.84,131072:3984.30,258048:9806.97}
for L in sorted(bd): items.append((f'DP TTFT {L2(L)}',dd[L],bd[L]))
for L in sorted(bp): items.append((f'PCP TTFT {L2(L)}',pp[L],bp[L]))
sb=[('SB DP C8',C['4_dp8_speed_bench_mix']['rows'][0]['input_tok_s'],30025.41),
    ('SB DP C64',C['4_dp8_speed_bench_mix']['rows'][1]['input_tok_s'],51941.43),
    ('SB PCP C8',C['7_pcp8_speed_bench_mix']['rows'][0]['input_tok_s'],44300.85),
    ('SB PCP C64',C['7_pcp8_speed_bench_mix']['rows'][1]['input_tok_s'],48251.88)]
items+=sb
names=[i[0] for i in items]; devs=[(i[1]-i[2])/i[2]*100 for i in items]
fig,ax=plt.subplots(figsize=(7.6,6.4))
cols=[GREEN if abs(v)<1 else (YELLOW if abs(v)<3 else RED) for v in devs]
ax.barh(range(len(devs)),devs,color=cols,height=.68)
ax.axvline(0,color='#202124',lw=1)
for i,v in enumerate(devs):
    ax.text(v+(0.06 if v>=0 else -0.06),i,f'{v:+.2f}%',va='center',
            ha='left' if v>=0 else 'right',fontsize=9,color='#3C4043')
ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=9.5); ax.invert_yaxis()
ax.set_xlabel('相对已发布基线的偏差 (%)'); ax.set_xlim(-3.2,3.4)
ax.axvspan(-1,1,color=GREEN,alpha=.07)
ax.set_title('20 项指标对账  ·  绿色 |偏差|<1%  ·  基线 commit 0be027b9 / 2026-08-26',
             fontsize=12,pad=12,loc='left')
ax.grid(axis='y',visible=False)
fig.tight_layout(); fig.savefig('c4_reproducibility.png',dpi=160); plt.close(fig)
print('四张图完成')
import statistics
print(f'20 项偏差: 最大 {max(devs,key=abs):+.2f}%  中位 {statistics.median([abs(v) for v in devs]):.2f}%  |偏差|<1% 的有 {sum(1 for v in devs if abs(v)<1)}/{len(devs)} 项')
