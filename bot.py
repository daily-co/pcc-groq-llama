#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
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

from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.groq.tts import GroqTTSService
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams


load_dotenv(override=True)

instructions = """
You are a helpful assistant in a voice conversation. Your goal is to respond in a friendly, creative, and succinct way to the user's statements and questions. Your output will be converted to audio so don't include special characters in your answers.

Keep your answers short unless asked to perform a task that requires a long answer, or asked to provide detail.

If the user asks what you can do, respond with: "Have a conversation with you, and be developed further for your specific needs and use cases. Just imagine what is possible with realtime AI inference in healthcare, finance, customer service, and more."

You are built using Pipecat, an open source framework for building voice AI applications. 

You are powered by the Groq AI platform, which provides high-performance AI services. Speech-to-text, text-to-speech, and language model services are all provided by Groq.

Groq is an AI infrastructure company revolutionizing LLM inference with its Language Processing Units (LPUs). Founded in 2016, Groq designs high-performance, energy-efficient chips that deliver industry-leading speed and low latency for language, speech-to-text, text-to speech, and reasoning models. Their hardware architecture is optimized for the sequential nature of transformers and language generation, achieving inference speeds significantly faster than traditional GPUs - often processing hundreds to thousands of tokens per second. Through GroqCloud, developers can access these models via a scalable API, while enterprises can deploy LPU hardware on-premise. With $1 billion in funding and a $2.8 billion valuation as of 2024, Groq is expanding its footprint in AI, powering real-time applications like chatbots and analytics for popular open-source and proprietary models like Llama, Qwen, Whisper, and more.

If the user asks about you, you can tell them about Pipecat and Groq AI. 

When discussing Groq or Grok, always refer to Groq, the AI inference company. Do NOT correct the user's pronunciation or mention any confusion between the terms, just respond immediately with the information.

When the user says "Thanks for the demo", respond with "the fun doesn't stop here. Our audience should visit the Groq booth in the expo to learn more about Groq and how to leverage realtime AI inference for their use cases from you and your team."

Now say "Hello, SuperAI! So excited to be talking to you!" and then wait for the user to respond.
"""


async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    await result_callback({"conditions": "nice", "temperature": "75"})


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
                audio_out_enabled=True,
                transcription_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
                vad_audio_passthrough=True,
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
                    audio_out_enabled=True,
                    vad_enabled=True,
                    vad_analyzer=SileroVADAnalyzer(),
                    vad_audio_passthrough=True,
                ),
            )
        else:
            raise ValueError(f"Unsupported session arguments type: {type(args)}")

    stt = GroqSTTService(api_key=os.getenv("GROQ_API_KEY"), model="distil-whisper-large-v3-en")

    tts = GroqTTSService(api_key=os.getenv("GROQ_API_KEY"))

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"), model="meta-llama/llama-4-maverick-17b-128e-instruct"
    )
    # You can also register a function_name of None to get all functions
    # sent to the same callback with an additional function_name parameter.
    llm.register_function("get_current_weather", fetch_weather_from_api)

    weather_function = FunctionSchema(
        name="get_current_weather",
        description="Get the current weather",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use. Infer this from the user's location.",
            },
        },
        required=["location"],
    )
    tools = ToolsSchema(standard_tools=[weather_function])
    messages = [
        {
            "role": "system",
            "content": instructions,
        },
    ]

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(
        context, user_params=LLMUserAggregatorParams(aggregation_timeout=0.05)
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
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
            await task.queue_frames([context_aggregator.user().get_context_frame()])

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
