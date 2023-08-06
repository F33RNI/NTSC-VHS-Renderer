"""
 Copyright (C) 2023 Fern Lane, NTSC-VHS-Renderer

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 See the License for the specific language governing permissions and
 limitations under the License.

 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY CLAIM, DAMAGES OR
 OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import logging
import multiprocessing
import os
import sys

import FramesProcessor
import GUI
import LoggingHandler
from JSONReaderWriter import load_json

# NTSC-VHS-Renderer version
__version__ = "1.0.0"

# Default config location
CONFIG_FILE = "config.json"


def parse_args():
    """
    Parses cli arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config.json file location",
                        default=os.getenv("NTSC_VHS_VIDEO_RENDERER_CONFIG_FILE", CONFIG_FILE))
    parser.add_argument("--version", action="version", version=__version__)
    return parser.parse_args()


def main() -> None:
    """
    Main entry
    :return:
    """
    # Multiprocessing fix for Windows
    if sys.platform.startswith("win"):
        multiprocessing.freeze_support()

    # Parse arguments
    args = parse_args()

    # Initialize logging and start listener as process
    logging_handler = LoggingHandler.LoggingHandler()
    logging_handler_process = multiprocessing.Process(target=logging_handler.configure_and_start_listener)
    logging_handler_process.start()
    LoggingHandler.worker_configurer(logging_handler.queue)
    logging.info("LoggingHandler PID: " + str(logging_handler_process.pid))

    # Log software version and GitHub link
    logging.info("NTSC-VHS-Renderer version: " + str(__version__))
    logging.info("https://github.com/F33RNI/NTSC-VHS-Renderer")

    # Load config with multiprocessing support
    if not os.path.exists(args.config):
        logging.error("File {} doesn't exist!".format(args.config))
        return
    config = multiprocessing.Manager().dict(load_json(args.config))

    try:
        # Initialize main processing class
        frames_processor = FramesProcessor.FramesProcessor(config, logging_handler.queue)

        # Load and open GUI in main process
        GUI.GUI(config, args.config, __version__, frames_processor, logging_handler.statusbar_queue)

        # "C:\\Users\\F3rni\\Videos\\ШКЯ - Как прожить жизнь так, чтобы на тебя не наорала бабка - Trim.mp4"
    except Exception as e:
        logging.error("Error running NTSC-VHS-Renderer!", exc_info=e)

    # If we're here, exit requested or error occurs
    logging.info("NTSC-VHS-Renderer exited successfully")

    # Finally, stop logging loop
    logging_handler.queue.put(None)


if __name__ == "__main__":
    main()
