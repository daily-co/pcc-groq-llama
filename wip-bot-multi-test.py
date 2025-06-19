#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import os
import re
import sys

from dataclasses import dataclass
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecatcloud.agent import (
    DailySessionArguments,
    SessionArguments,
    WebSocketSessionArguments,
)
from pipecat.frames.frames import (
    Frame,
    EndFrame,
    SystemFrame,
    LLMTextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.groq.tts import GroqTTSService
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams


load_dotenv(override=True)


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

    # Pick strings for our language tags that are single tokens and not expected
    # in normal text output.
    EN_TAG = "EN"
    AR_TAG = "AR"

    # defin a Segment data type that is a tuple of (language, text)
    # use @dataclass to define the Segment data type
    @dataclass
    class Segment:
        language: str
        text: list[LLMTextFrame]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffering = False
        self.segments: list[LanguageTagger.Segment] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            # Clear the segments buffer. We will send LLMFullResponseStartFrames with each language segment.
            self.segments = []
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            await self.flush_language_segments()
            # Don't push the LLMFullResponseEndFrame; we'll push it as part of each sequence
            return

        if isinstance(frame, NextLanguageSequenceFrame):
            await self.flush_language_segments()

        if isinstance(frame, LLMTextFrame):
            logger.debug(f"!!! LLMTextFrame: {frame.text}")
            match = re.match(r"(.*)(" + self.EN_TAG + r"|" + self.AR_TAG + r")(.*)", frame.text)
            if match:
                pre_text = match.group(1)
                language = match.group(2)
                post_text = match.group(3)
                if pre_text.strip():
                    await self.push_text(pre_text)
                self.segments.append(LanguageTagger.Segment(language=language, text=[]))
                await self.push_text(LanguageTagFrame(language=language))
                if post_text.strip():
                    await self.push_text(post_text)
                return
            else:
                await self.push_text(frame)
                return

        await self.push_frame(frame, direction)

    async def maybe_push_frame(self, frame: Frame, direction: FrameDirection):
        pass

    async def push_text(self, text_or_frame: str | LLMTextFrame):
        # We expect to always get a language tag at the start of a response. We prompt
        # the LLM to try to make that happen. But, of course, it might not. So if there was
        # no initial language tag, we might need to create a segment here.
        if not self.segments:
            self.segments.append(LanguageTagger.Segment(language=self.EN_TAG, text=[]))
        if isinstance(text_or_frame, str):
            self.segments[-1].text.append(LLMTextFrame(text=text_or_frame))
        else:
            self.segments[-1].text.append(text_or_frame)

    async def flush_language_segments(self):
        if not self.segments:
            return
        segment = self.segments.pop(0)
        await self.push_frame(LanguageTagFrame(language=segment.language))
        await self.push_frame(LLMFullResponseStartFrame())
        for text_frame in segment.text:
            await self.push_frame(text_frame)
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


class TTSSegmentSequencer(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseEndFrame):
            logger.debug("!!! pushing NextLanguageSequenceFrame upstream")
            await self.push_frame(NextLanguageSequenceFrame(), direction=FrameDirection.UPSTREAM)
        await self.push_frame(frame, direction)


instructions = f"""
You are a helpful assistant in a voice conversation. Your goal is to respond in a friendly, creative, and succinct way to the user's statements and questions. Your output will be converted to audio so don't include special characters in your answers.

Keep your answers short unless asked to perform a task that requires a long answer, or asked to provide detail.

You understand both English and Arabic. You can respond in either language.

# Rules for responding

- By default, respond in the language the user used most recently.
- Follow user instructions to switch languages.
- Translate between languages as requested or as appropriate.

# Rules for formatting

## At the beginning of each response, prepend the language you are using as either {LanguageTagger.EN_TAG} or {LanguageTagger.AR_TAG}.

Example:

{LanguageTagger.EN_TAG}
How are you today?

## Whenever you switch languages while responding, insert the language tag {LanguageTagger.EN_TAG} or {LanguageTagger.AR_TAG}.

Example switching between languages 1:

{LanguageTagger.EN_TAG}
How are you today? Translates as ...
{LanguageTagger.AR_TAG}
مرحباً، سعيدٌ بكونك هنا!

Example switching between languages 2:

{LanguageTagger.EN_TAG}
I will count in both languages, alternating between them.
two,
{LanguageTagger.AR_TAG}
ثلاثة,
{LanguageTagger.EN_TAG}
four
{LanguageTagger.AR_TAG}
خمسة,
{LanguageTagger.EN_TAG}
six
{LanguageTagger.AR_TAG}
سبعة,
"""


class TestSTTService(GroqSTTService):
    def language_to_service_language(self, language: str) -> str:
        logger.info("Setting whisper to multi-lingual (no language specified)")
        return ""


async def main(args: SessionArguments):
    if isinstance(args, WebSocketSessionArguments):
        logger.debug("Starting WebSocket bot")

        start_data = args.websocket.iter_text()
        await start_data.__anext__()
        call_data = json.loads(await start_data.__anext__())
        stream_sid = call_data["start"]["streamSid"]
        transport = FastAPIWebsocketTransport(
            websocket=args.websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.4)),
                vad_audio_passthrough=True,
                serializer=TwilioFrameSerializer(stream_sid),
            ),
        )
    elif isinstance(args, DailySessionArguments):
        logger.debug("Starting Daily bot")
        transport = DailyTransport(
            args.room_url,
            args.token,
            "Respond bot",
            DailyParams(
                audio_in_enabled=True,
                audio_in_passthrough=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )
    else:
        from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection
        from pipecat.transports.base_transport import TransportParams
        from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

        if isinstance(args, SmallWebRTCConnection):
            logger.debug("Starting SmallWebRTC bot")
            transport = SmallWebRTCTransport(
                webrtc_connection=args,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_in_passthrough=True,
                    audio_out_enabled=True,
                    vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
                ),
            )
        else:
            raise ValueError(f"Unsupported session arguments type: {type(args)}")

    stt = TestSTTService(
        api_key=os.getenv("GROQ_API_KEY"), model="whisper-large-v3-turbo", language="multi"
    )

    tts_en = GroqTTSService(
        api_key=os.getenv("GROQ_API_KEY"), model_name="playai-tts", voice_id="Calum-PlayAI"
    )
    tts_ar = GroqTTSService(
        api_key=os.getenv("GROQ_API_KEY"), model_name="playai-tts-arabic", voice_id="Nasser-PlayAI"
    )

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"), model="meta-llama/llama-4-maverick-17b-128e-instruct"
    )

    language_gate_en = LanguageGate(language="EN")
    language_gate_ar = LanguageGate(language="AR")
    language_tagger = LanguageTagger()
    tts_sequencer = TTSSegmentSequencer()

    messages = [
        {
            "role": "system",
            "content": instructions,
        },
        # {
        #     "role": "user",
        #     "content": "Ask me how I am doing today, and then translate what you asked into Arabic.",
        # },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(
        context, user_params=LLMUserAggregatorParams(aggregation_timeout=0.05)
    )

    class showme(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if (
                isinstance(frame, TTSStartedFrame)
                or isinstance(frame, TTSStoppedFrame)
                or isinstance(frame, LLMFullResponseEndFrame)
            ):
                logger.debug(f"!!! {self.__class__.__name__}: {frame}")
            await self.push_frame(frame, direction)

    class showme_en(showme):
        pass

    class showme_ar(showme):
        pass

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            language_tagger,
            ParallelPipeline(
                [language_gate_en, tts_en, showme_en()],
                [language_gate_ar, tts_ar, showme_ar()],
            ),
            tts_sequencer,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    if isinstance(args, DailySessionArguments):

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.cancel()
    else:

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            # Kick off the conversation.
            await task.queue_frames(
                [
                    context_aggregator.user().get_context_frame(),
                ]
            )

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()

        @transport.event_handler("on_client_closed")
        async def on_client_closed(transport, client):
            logger.info("Client closed connection")
            await task.cancel()

    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    await runner.run(task)


async def bot(args: SessionArguments):
    try:
        await main(args)
        logger.info("Bot process completed")
    except Exception as e:
        logger.exception(f"Error in bot process: {str(e)}")
        raise


if __name__ == "__main__":
    from local_development_runner import local_development_main

    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    logger.info("Starting local development mode")
    local_development_main()
