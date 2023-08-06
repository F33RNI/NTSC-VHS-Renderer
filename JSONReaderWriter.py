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

import json
import logging
import multiprocessing.managers
import os.path
from multiprocessing.managers import DictProxy


def load_json(file_name: str, logging_enabled=True):
    """
    Loads json from file_name
    :param file_name: filename to load
    :param logging_enabled: set True to have logs
    :return: json if loaded or None if not
    """
    try:
        if os.path.exists(file_name):
            if logging_enabled:
                logging.info("Loading {0}".format(file_name))

            messages_file = open(file_name, encoding="utf-8")
            json_content = json.load(messages_file)
            messages_file.close()

            if json_content is not None:
                if logging_enabled:
                    logging.info("Loaded json from {0}".format(file_name))
            else:
                if logging_enabled:
                    logging.error("Error loading json data from file {0}".format(file_name))
                return None
        else:
            if logging_enabled:
                logging.warning("No {0} file! Returning empty json".format(file_name))
            return None

    except Exception as e:
        if logging_enabled:
            logging.error("Error loading json data from file {0}".format(file_name), exc_info=e)
        return None

    return json_content


def save_json(file_name: str, content, logging_enabled=True):
    """
    Saves
    :param file_name: filename to save
    :param content: JSON dictionary
    :param logging_enabled: set True to have logs
    :return:
    """
    if logging_enabled:
        logging.info("Saving to {0}".format(file_name))
    file = open(file_name, "w")
    if type(content) == dict:
        json.dump(content, file, indent=4)
    elif type(content) == DictProxy:
        json.dump(content.copy(), file, indent=4)
    file.close()
