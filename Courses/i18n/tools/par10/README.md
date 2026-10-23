# par10：中英版式对比十轮的工具

手册见 `构建手册/12-翻译成英文.md` 第七节。用法：

```bash
cd Courses/i18n/tools/par10
python3 mk.py topic-0N /tmp/t0Nen R1 1280 390          # 出材料：zh-ref、tmp.en、并排切片、layout、lines-W.txt、overlap.txt → $WORK/par/R1/
python3 prompt.py topic-0N /tmp/t0Nen R1 "00,01"       # 按 COMMON.md + FOCUS.md 生成这一组的任务书 → $WORK/par/R1/PROMPT-0001.md
python3 lines.py ZH.html EN.html 390 [--folded]        # 逐块行数对比（a/span 互换算同构，标签错位后自动重新对齐）
I18N_WORK=/tmp/t0Nen python3 shots.py                  # R3：每张图中英各截一张 → par/R3/figs/
I18N_WORK=… I18N_TOPIC=… I18N_EN=… python3 sbs_folded.py 1280   # R4：折叠态并排切片
python3 thumbs.py $WORK/par/R10 1280                   # R10：整页缩略图并排
python3 floor_en_topic05.py                            # 讲义字数表按英文词数重算（各专题各一份，路径写在脚本头）
```
