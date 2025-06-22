import re

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    EndFrame,
    SystemFrame,
)
from loguru import logger
from dataclasses import dataclass
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext


@dataclass
class LanguageTagFrame(Frame):
    """Frame that indicates the language of the following text"""

    language: str


@dataclass
class NextLanguageSequenceFrame(Frame):
    pass


class LanguageTagger(FrameProcessor):
    """Frame processor to remove single-token language tags from the LLM
    output stream, buffer text segments, and emit text segments with language tags."""

    @dataclass
    class Segment:
        language: str
        text: str
        buffered: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_language = None
        self.segments: list[LanguageTagger.Segment] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            # Don't automatically push this frame
            self.segments = []
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            # Don't automatically push this frame
            self.current_language = None
            await self.flush_language_segment()
            return

        if isinstance(frame, NextLanguageSequenceFrame):
            # We expect this frame to come upstream to us from the TTSSegmentSequencer
            await self.flush_language_segment()

        if isinstance(frame, LLMTextFrame):
            # logger.debug(f"!!! LLMTextFrame: {frame}")
            match = re.match(r"(.*)(EN|AR)(.*)", frame.text)
            if match:
                pre_text = match.group(1)
                if pre_text:
                    await self.push_text(pre_text)
                language = match.group(2)
                if language and language != self.current_language:
                    self.current_language = language
                    await self.create_segment(language)
                post_text = match.group(3)
                if post_text:
                    await self.push_text(post_text)
            else:
                await self.push_text(frame)
            return

        await self.push_frame(frame, direction)

    async def create_segment(self, language: str):
        should_buffer = len(self.segments) > 0
        logger.debug(f"Creating segment: {language}, should_buffer: {should_buffer}")
        self.segments.append(
            LanguageTagger.Segment(language=language, text="", buffered=should_buffer)
        )
        if not should_buffer:
            await self.push_frame(LanguageTagFrame(language=language))
            await self.push_frame(LLMFullResponseStartFrame())

    async def push_text(self, text_or_frame: str | LLMTextFrame):
        # We expect to always get a language tag at the start of a response. We prompt
        # the LLM to try to make that happen. But, of course, it might not. So if there was
        # no initial language tag, we might need to create a segment here.
        if not self.segments:
            await self.create_segment("EN")
        frame = (
            text_or_frame
            if isinstance(text_or_frame, LLMTextFrame)
            else LLMTextFrame(text=text_or_frame)
        )
        if not self.segments[-1].buffered:
            await self.push_frame(frame)
        else:
            self.segments[-1].text += frame.text

    async def flush_language_segment(self):
        if not self.segments:
            return
        segment = self.segments.pop(0)
        if not segment.buffered:
            await self.push_frame(LLMFullResponseEndFrame())
            return
        await self.push_frame(LanguageTagFrame(language=segment.language))
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(text=segment.text))
        await self.push_frame(LLMFullResponseEndFrame())


class LanguageGate(FrameProcessor):
    """Frame processor that opens and closes a pipeline based on language tags.
    This directs the language-specific LLM response segments to the correct TTS element."""

    def __init__(self, language, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.language = language
        self.open = False

    def _should_passthrough_frame(self, frame):
        if self.open:
            return True
        return isinstance(frame, (EndFrame, SystemFrame))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LanguageTagFrame):
            if frame.language == self.language:
                self.open = True
            else:
                self.open = False

        if self._should_passthrough_frame(frame):
            await self.push_frame(frame, direction)


class LanguageRetagger(FrameProcessor):
    def __init__(self, language, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.language = language

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            await self.push_frame(frame)
            await self.push_frame(LLMTextFrame(text=f"{self.language}\n"))
        else:
            await self.push_frame(frame, direction)


class TTSSegmentSequencer(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseEndFrame):
            await self.push_frame(NextLanguageSequenceFrame(), direction=FrameDirection.UPSTREAM)
        await self.push_frame(frame, direction)


class MultiContext(OpenAILLMContext):
    """Quick and dirty re-concatenation of messages with the same role. In this application, we'll mostly see this from multi-lingual responses that we split for the separate TTS pipelines to process."""

    def add_message(self, message):
        logger.debug(f"!!! add_message: {message}")
        if self._messages[-1]["role"] == message["role"]:
            self._messages[-1]["content"] += "\n\n" + message["content"]
        else:
            self._messages.append(message)
