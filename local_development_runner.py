#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import importlib.util
import logging
import os
import sys
from contextlib import asynccontextmanager
from inspect import iscoroutinefunction
from typing import Any, Callable, Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

# Load environment variables
load_dotenv(override=True)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("pipecat-server")
# We have to explicitly set the new logger level to INFO
logger.setLevel(logging.INFO)

app = FastAPI()

# Store connections by pc_id
pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = ["stun:stun.l.google.com:19302"]

# Mount the frontend at /
app.mount("/client", SmallWebRTCPrebuiltUI)

# Store the bot module and function info
bot_module: Any = None
run_bot_func: Optional[Callable] = None


def import_bot_file(file_path: str) -> Any:
    """Dynamically import the bot file and determine how to run it.

    Returns:
        module: The imported module
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Bot file not found: {file_path}")

    # Extract module name without extension
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Fall back to main function
    if hasattr(module, "main") and iscoroutinefunction(module.main):
        return module

    raise AttributeError(f"No run_bot or async main function found in {file_path}")


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    if not run_bot_func:
        return {"error": "No bot function available to run"}

    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"], type=request["type"], restart_pc=request.get("restart_pc", False)
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # We've already checked that run_bot_func exists
        assert run_bot_func is not None
        background_tasks.add_task(run_bot_func, pipecat_connection)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.close() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


async def run_standalone_bot() -> None:
    """Run a standalone bot that doesn't require WebRTC"""
    global run_bot_func
    if run_bot_func is not None:
        await run_bot_func()
    else:
        raise RuntimeError("No bot function available to run")


def local_development_main():
    host = "localhost"
    port = 7860
    # Get the __file__ of the script that called main()
    import inspect

    caller_frame = inspect.stack()[1]
    caller_globals = caller_frame.frame.f_globals
    bot_file = caller_globals.get("__file__")

    if not bot_file:
        print("❌ Could not determine the bot file. Pass it explicitly to main().")
        sys.exit(1)

    # Import the bot file
    try:
        global run_bot_func, bot_module
        bot_module = import_bot_file(bot_file)
        run_bot_func = bot_module.main
        logger.info(f"Successfully loaded bot from {bot_file}, starting web server...")
        uvicorn.run(app, host=host, port=port)

    except Exception as e:
        logger.error(f"Error loading bot file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    local_development_main()
