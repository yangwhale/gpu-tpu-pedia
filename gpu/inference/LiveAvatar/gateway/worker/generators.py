"""VideoGenerator 实现。

LiveKit 的协议只有三个方法（livekit/agents/voice/avatar/_types.py）：
    push_audio(frame | AudioSegmentEnd)   喂音频
    clear_buffer()                        打断时立刻停
    __aiter__()                           持续吐出视频帧和音频帧

⚠️ clear_buffer 是三个里最要命的。人已经不说话了而屏幕上的嘴还在动，
   恐怖谷一下就掉进去了。两个实现都必须把在途缓冲丢干净。
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Union

import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, VideoGenerator

AVOut = Union[rtc.VideoFrame, rtc.AudioFrame, AudioSegmentEnd]


class StaticImageGenerator(VideoGenerator):
    """P0 用：静帧 ＋ 原音频回放。

    不做任何生成 —— 它存在的唯一目的是**验证契约**：
    真 agent 用标准 AvatarSession 调进来，房间里能看到形象、能听到声音、
    打断能停。契约错了在这一步暴露，成本最低。
    """

    def __init__(self, image_rgba: np.ndarray, *, fps: int = 25):
        self._img = np.ascontiguousarray(image_rgba, dtype=np.uint8)
        self._h, self._w = self._img.shape[:2]
        self._fps = fps
        self._q: asyncio.Queue[AVOut] = asyncio.Queue()
        self._audio_q: asyncio.Queue[rtc.AudioFrame | AudioSegmentEnd] = asyncio.Queue()
        self._video_task: asyncio.Task | None = None

    @property
    def size(self) -> tuple[int, int]:
        return self._w, self._h

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        await self._audio_q.put(frame)

    def clear_buffer(self) -> None:
        for q in (self._q, self._audio_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

    def _frame(self) -> rtc.VideoFrame:
        return rtc.VideoFrame(
            width=self._w, height=self._h,
            type=rtc.VideoBufferType.RGBA, data=self._img.tobytes(),
        )

    async def __aiter__(self) -> AsyncIterator[AVOut]:
        interval = 1.0 / self._fps
        next_at = asyncio.get_running_loop().time()
        while True:
            # 音频优先：有多少发多少，不阻塞
            while not self._audio_q.empty():
                yield self._audio_q.get_nowait()
            yield self._frame()
            next_at += interval
            delay = next_at - asyncio.get_running_loop().time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_at = asyncio.get_running_loop().time()   # 落后就重新对齐，不追债


class LiveAvatarGenerator(VideoGenerator):
    """P1 用：接真正的 LiveAvatar。

    还没实现 —— 主要工作量在音频流式化：上游的
    ``get_audio_embed_bucket_fps`` 要**整段音频**才能算出 num_repeat，
    而我们这里音频是一块一块来的。

    改造路径（已确认架构上可行）：生成循环里每个 block 只用
    ``audio_input[..., left_idx:right_idx]`` 一个切片，所以可以改成
    「来一块编一块，还有音频就多转一轮」，不必预知总长。
    出帧端用 ``causal_s2v_pipeline_tpp_blockwise`` 里那个 ``yield``。

    实测最优参数：384×256、TRIM_K=4、单卡 1.357× 实时、抖动近零、
    首帧 1.26 s（该值压不下去，见 OPTIMIZATION-JOURNAL.md）。
    """

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "P1 未实现：需要先把上游的音频编码改成可分块喂入。"
            "详见 gateway/README.md 的路线图。"
        )
