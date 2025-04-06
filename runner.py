#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os
import time
from typing import Optional

import aiohttp
from loguru import logger
from pipecat.transports.services.helpers.daily_rest import (
    DailyRESTHelper,
    DailyRoomParams,
    DailyRoomProperties,
)


async def configure(aiohttp_session: aiohttp.ClientSession):
    (url, token, _) = await configure_with_args(aiohttp_session)
    return (url, token)


async def configure_with_args(
    aiohttp_session: aiohttp.ClientSession, parser: Optional[argparse.ArgumentParser] = None
):
    if not parser:
        parser = argparse.ArgumentParser(description="Daily AI SDK Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=False, help="URL of the Daily room to join"
    )
    parser.add_argument(
        "-k",
        "--apikey",
        type=str,
        required=False,
        help="Daily API Key (needed to create an owner token for the room)",
    )

    args, unknown = parser.parse_known_args()
    key = args.apikey or os.getenv("DAILY_API_KEY")
    daily_rest_helper = DailyRESTHelper(
        daily_api_key=key,
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    url = (
        args.url
        or (
            # Create a Daily room with a 10 minute expiration and no prejoin UI
            await daily_rest_helper.create_room(
                DailyRoomParams(
                    properties=DailyRoomProperties(
                        exp=time.time() + 600, start_video_off=True, enable_prejoin_ui=False
                    )
                )
            )
        ).url
    )

    if not url:
        raise Exception(
            "No Daily room specified. use the -u/--url option from the command line, or set DAILY_SAMPLE_ROOM_URL in your environment to specify a Daily room URL."
        )

    if not key:
        raise Exception(
            "No Daily API key specified. use the -k/--apikey option from the command line, or set DAILY_API_KEY in your environment to specify a Daily API key, available from https://pipecat.daily.co/<your-org-id>/settings/daily."
        )

    # Create a meeting token for the given room with an expiration 1 hour in
    # the future.
    expiry_time: float = 60 * 60

    token = await daily_rest_helper.get_token(url, expiry_time)
    logger.info(f"Daily room URL: {url}")
    return (url, token, args)
