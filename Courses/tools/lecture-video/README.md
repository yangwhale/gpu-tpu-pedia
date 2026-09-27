# 讲课视频：录音 ＋ 课件 → mp4

录音是主资产（`tools/topic05-audio/topic05-lecture-s<N>.mp3` 对应的现场原声），视频只是给它配画面：讲到哪，课件翻到哪、框到哪。

1. `align.py`：faster-whisper large-v3 给录音打逐字时间戳 → `topic05-audio/s<N>-words.json`（CPU 约 1× 实时）
2. `srt.py`：原文字稿 ＋ 逐字时间 → 句级字幕 `topic05-audio/s<N>.srt`（文字用原稿，时间用识别）
3. `s<N>-cues.json`：提示表——哪句台词（`at`，开头几个字）对应页面哪个元素（`target`）、哪一块（`sub`，按元素宽高的比例）、怎么标（`box`／`circle`／`under`／`none`）、放哪段动画（`play`）
4. `resolve.py`：把 `at` 换成秒数；`render.py`：无头 Chromium 逐帧确定性渲染（页面动画按时间设 currentTime），分片并行、合音

第一节：8 分 21 秒，12 片并行渲染约 2 分半。
