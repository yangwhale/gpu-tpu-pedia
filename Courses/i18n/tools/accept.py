"""把 check.py 报的超宽标签登记进白名单（先渲染确认放得下再登记）。

    I18N_WORK=<工作区> I18N_TOPIC=topic-NN python3 accept.py NN "理由"

登记到 $I18N_WORK/rounds/ACCEPTED-WIDTH.md（check.py 读同一个文件），每行：块号 | #序号 | 中文 | 英文 | 宽度比 | 理由。
⚠️ 它会把这一块**当前所有**超宽标签一次登记进去 —— 只在逐个看过渲染之后跑。
"""
import subprocess, sys, re, os, pathlib
WORK = os.environ['I18N_WORK']
n, why = sys.argv[1], sys.argv[2]
chk = pathlib.Path(__file__).resolve().parent / 'check.py'
out = subprocess.run(['python3', str(chk), f'{n}.zh.html', f'{n}.en.html', '--render'],
                     cwd=WORK + '/chunks', capture_output=True, text=True, env=dict(os.environ)).stdout
rows = re.findall(r'#(\d+)\s+([\d.]+)×\s+zh「(.*?)」\s+en "(.*?)"', out)
with open(WORK + '/rounds/ACCEPTED-WIDTH.md', 'a') as f:
    for i, r, z, e in rows:
        f.write(f'{n} | #{i} | {z} | {e} | {r} | {why}\n')
print(len(rows), 'accepted')
