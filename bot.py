#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import os
import sys

import aiohttp
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
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.groq.tts import GroqTTSService
from pipecat.transports.base_transport import BaseTransport
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

from runner import configure

load_dotenv(override=True)

instructions = """
You are a helpful assistant in a voice conversation. Your goal is to respond in a friendly, creative, and succinct way to the user's statements and questions. Your output will be converted to audio so don't include special characters in your answers.

Keep your answers short unless asked to perform a task that requires a long answer, or asked to provide detail.

You have access to a function get_current_weather that you can use to look up the current weather in a location. get_current_weather is a demonstration function that always returns the same information for every location. If the user expresses confusion about the weather information you provide, tell them that everything is working as expected, and the get_current_weather function is a demonstration function that always returns hard-coded data.

If the user asks what you can do, you can respond that you can have a conversation with them and that you have access to current weather information anywhere in the world.

Now say "Hi, nice to be talking to you!" and then wait for the user to respond.
"""


async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    await result_callback({"conditions": "nice", "temperature": "75"})


async def main(transport: BaseTransport, args: SessionArguments):
    stt = GroqSTTService(api_key=os.getenv("GROQ_API_KEY"), model="distil-whisper-large-v3-en")

    tts = GroqTTSService(api_key=os.getenv("GROQ_API_KEY"))

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"), model="meta-llama/llama-4-scout-17b-16e-instruct"
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
    context_aggregator = llm.create_context_aggregator(context)

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

    if isinstance(args, WebSocketSessionArguments):

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected: {client}")
            # Kick off the conversation
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected: {client}")
            await task.cancel()
    elif isinstance(args, DailySessionArguments):

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.cancel()

    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    await runner.run(task)


async def bot(args: SessionArguments):
    try:
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
        await main(transport, args)
        logger.info("Bot process completed")
    except Exception as e:
        logger.exception(f"Error in bot process: {str(e)}")
        raise


async def local():
    logger.info("Starting local bot")
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    async with aiohttp.ClientSession() as session:
        if os.getenv("DAILY_API_KEY"):
            (room_url, token) = await configure(session)

            logger.debug("Starting Daily bot")
            args = DailySessionArguments(
                session_id=None,
                room_url=room_url,
                token=token,
                body=None,
            )
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
            await main(transport, args)
        else:
            logger.error("DAILY_API_KEY must be set in your .env file for local testing.")


print("Module name:", __name__)
if __name__ == "__main__":
    print("Running bot.py as main script")
    asyncio.run(local())
