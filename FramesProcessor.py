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

import logging
import multiprocessing
import os
import shutil
import subprocess
import threading
import time
from ctypes import c_char_p, c_bool, c_int32

import numpy as np
import psutil as psutil
from PIL import Image, ImageEnhance

from LoggingHandler import worker_configurer

# Number of preloaded frames onto disk
# Ex. (50 x 1920 x 1080 x 24) / 8 / 1024 / 1024 = ~297MB @ FullHD (~1.7s @ 30fps)
# Temp folder might be this size x4 or more
# Must be >2 frames
BUFFER_SIZE_FRAMES = 50

# Filter K of preview FPS to smooth values
PREVIEW_FPS_FILTER_K = 0.98


def frames_to_time_str(current_frame: int, frames_total: int, fps: float) -> str:
    """
    Converts frames to string info about time
    :param current_frame: 1 - N
    :param frames_total: N
    :param fps: 0-...
    :return: hh:mm:ss.SSS
    """
    if fps == 0:
        return "-"
    frames_diff = frames_total - current_frame
    seconds_total = frames_diff / fps
    hours = int(seconds_total // 3600)
    minutes = int((seconds_total - (hours * 3600)) // 60)
    seconds = (seconds_total - (hours * 3600)) - (minutes * 60)
    return "{:02d}:{:02d}:{:06.3f}".format(hours, minutes, round(seconds, 3))


def search_for_executable(name: str) -> str:
    """
    Tries to find executable in program folder / subfolders. Otherwise, raises Exception
    :param name: name of executable to search (ex. ffmpeg)
    :return:
    """
    for dir_path, dir_names, file_names in os.walk("."):
        for filename in [f for f in file_names if os.path.splitext(f)[0] == name]:
            executable_path = os.path.join(dir_path, filename)
            logging.info("{0} path: {1}".format(name, executable_path))
            return executable_path
    raise Exception("Cannot find {0} executable in program folder!".format(name))


def prepare_empty_frames(frames: int):
    """
    Creates empty bmp files for each frame
    :param frames:
    :return:
    """
    # Create frames directory if not exists
    frames_dir = os.path.join("temp", "frames")
    if not os.path.exists(frames_dir):
        logging.info("Creating temp directories")
        os.makedirs(frames_dir)

    # Generate empty BMP files
    logging.info("Generating {} empty frames".format(frames))
    for i in range(1, frames + 1):
        open(os.path.join(frames_dir, str(i).zfill(6) + ".bmp"), "w").close()


def _run_command_and_capture_output(command: list) -> str:
    """
    Executes subprocess.Popen and waits for output
    :param command: command to execute with arguments as list
    :return: command result
    """
    logging.info("Running {}".format(" ".join(command)))
    command_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return command_process.communicate()[0].decode("utf-8")


def delete_frames(frames_from: int, frames_to: int, input_=True) -> None:
    """
    Deletes frames from frames or output directory
    :param frames_from: 1-N
    :param frames_to: 1-N
    :param input_: True for frames/, False for output/
    :return:
    """
    # Check frames dir
    frames_dir = os.path.join("temp", "frames" if input_ else "output")
    if not os.path.exists(frames_dir):
        return

    # Log info
    logging.info("Removing {0} frames from {1} to {2}".format(frames_dir, frames_from, frames_to))

    # Scan all files and delete if they are in range
    for file in os.listdir(frames_dir):
        if os.path.splitext(file)[1].lower() == ".bmp":
            try:
                frame_number = int(os.path.splitext(file)[0].strip())
                if frames_to >= frame_number >= frames_from:
                    os.remove(os.path.join(frames_dir, file))
            except:
                pass


class FramesProcessor:
    def __init__(self, config: dict, logging_queue: multiprocessing.Queue):
        self.config = config
        self.logging_queue = logging_queue

        # Executables
        self.ffmpeg_path = multiprocessing.Manager().Value(c_char_p, search_for_executable("ffmpeg"))
        self.ffprobe_path = multiprocessing.Manager().Value(c_char_p, search_for_executable("ffprobe"))
        self.ntsc_video_path = multiprocessing.Manager().Value(c_char_p, search_for_executable("ntsc_video"))
        self.ntsc_video_vhs_path = multiprocessing.Manager().Value(c_char_p, search_for_executable("ntsc_video_vhs"))

        # Rendering process controls
        self.rendering_pause_request = multiprocessing.Manager().Value(c_bool, False)
        self.rendering_resume_request = multiprocessing.Manager().Value(c_bool, False)
        self.rendering_single_frame_forward_request = multiprocessing.Manager().Value(c_bool, False)
        self.rendering_stop_request = multiprocessing.Manager().Value(c_bool, False)
        self.rendering_process_active = multiprocessing.Manager().Value(c_bool, False)
        self.rendering_process_paused = multiprocessing.Manager().Value(c_bool, False)

        # Queue of preview images (must be used in GUI.py)
        # put(None) -> Exit, put(0) -> Rendering finished
        self.preview_queue = multiprocessing.Queue(-1)

        # Playback information
        self.playback_info = multiprocessing.Manager().Value(c_char_p, "")
        self.rendering_progress = multiprocessing.Manager().Value(c_int32, 0)

        # Currently processing frame
        self.current_frame_absolute = multiprocessing.Manager().Value(c_int32, 0)
        self.frames_max = multiprocessing.Manager().Value(c_int32, 0)

        # NTSC-CRT process (MUST BE USED ONLY INSIDE RENDERING PROCESS)
        self._ntsc_process = None
        self._file_scan_lock = multiprocessing.Manager().Lock()

        # List of ready-to-use frames (FOR NTSC AND RENDERING PROCESS ONLY)
        self._extracted_frames = multiprocessing.Manager().list()
        self._processed_frames = multiprocessing.Manager().list()

    def start_rendering(self, file: str, frame_from: int, frame_to: int, video_parameters: dict,
                        render_to_file: str | None):
        """
        Starts rendering process
        :param file:
        :param frame_from:
        :param frame_to:
        :param video_parameters:
        :param render_to_file:
        :return:
        """
        if self.rendering_process_active.value:
            logging.error("Cannot start rendering process because it is already started!")
            return

        # Start rendering process
        rendering_process = multiprocessing.Process(target=self._rendering_process, args=(file,
                                                                                          frame_from,
                                                                                          frame_to,
                                                                                          video_parameters,
                                                                                          render_to_file,))
        rendering_process.start()
        logging.info("Rendering process PID: {}".format(rendering_process.pid))

        # Wait until started
        logging.info("Waiting for process to start")
        while not self.rendering_process_active.value:
            time.sleep(0.01)
        logging.info("Rendering started")

    def _rendering_process(self, file: str,
                           frame_from: int,
                           frame_to: int,
                           video_parameters: dict,
                           render_to_file: str | None):
        """
        Main rendering process
        :param file: Input file
        :param frame_from: 1-N
        :param frame_to: 1-N
        :param video_parameters:
        :param render_to_file: None for preview
        :return:
        """
        # Setup logging for current process
        worker_configurer(self.logging_queue)

        rendering_finished = False
        try:
            # Reset flags and variables
            self.rendering_process_paused.value = False
            self.rendering_pause_request.value = False
            self.rendering_resume_request.value = False
            self.rendering_single_frame_forward_request.value = False
            self.rendering_stop_request.value = False
            self.playback_info.value = ""
            self.rendering_progress.value = 0
            self.current_frame_absolute.value = frame_from
            self.frames_max.value = frame_to

            # Delete temp directory if it exists with all files inside it
            if os.path.exists("temp"):
                logging.info("Deleting temp directory")
                shutil.rmtree("temp")

            # Clear lists
            while len(self._extracted_frames) > 0:
                self._extracted_frames.pop()
            while len(self._processed_frames) > 0:
                self._processed_frames.pop()

            # Extract first batch of images
            self.extract_frames(file, frame_from, min(frame_from + BUFFER_SIZE_FRAMES, frame_to))

            # Start ntsc process
            self._ntsc_processor_start(frame_to - frame_from + 1, video_parameters)

            # Get output file format and codec
            format_config = self.config["out_format"]
            file_format = "mp4" if format_config == 0 else ("mov" if format_config == 1 else "jpg")
            codec = "libx264" if (format_config == 0 or format_config == 1) else "mjpeg"
            render_temp_file = os.path.join("temp",
                                            ("render." + file_format) if (format_config == 0 or format_config == 1)
                                            else os.path.join("render", "%06d.jpg"))

            # Create render folder (in jpeg mode)
            if format_config == 2 and not os.path.exists(os.path.join("temp", "render")):
                os.makedirs(os.path.join("temp", "render"))

            # Get output bitrate
            bitrate = 0
            if self.config["out_bitrate_equal_to_input"]:
                if int(video_parameters["bit_rate"]) > 0:
                    bitrate = int(video_parameters["bit_rate"])
            else:
                if int(self.config["out_bitrate"]) > 0:
                    bitrate = int(self.config["out_bitrate"]) * 1000

            # Rendering loop
            ntsc_paused_too_fast = False
            frames_dir_output = os.path.join("temp", "output")
            frames_dir_input = os.path.join("temp", "frames")
            time_started = 0
            fps_filtered = 0
            file_started = False
            ffmpeg_process = None
            current_frame = 1
            bloom_initialized = False
            vignette = None
            vignette_strength = 0
            while True:
                # Set active flag
                self.rendering_process_active.value = True

                # Check if we have current frame, and we have next frame or ntsc process finished (to avoid handling)
                if current_frame in self._processed_frames \
                        and (current_frame + 1 in self._processed_frames or self._ntsc_process is None):
                    try:
                        # Construct filename
                        filename = "{:06d}.bmp".format(current_frame)

                        # Load processed frame as RGB image
                        frame = Image.open(os.path.join(frames_dir_output, filename)).convert("RGB")

                        # Apply brightness
                        if self.config["brightness_enabled"] and int(self.config["brightness"]) != 0:
                            frame = ImageEnhance.Brightness(frame) \
                                .enhance((int(self.config["brightness"]) / 100.) + 1.)

                        # Apply contrast
                        if self.config["contrast_enabled"] and int(self.config["contrast"]) != 0:
                            frame = ImageEnhance.Contrast(frame) \
                                .enhance((int(self.config["contrast"]) / 100.) + 1.)

                        # Apply saturation
                        if self.config["saturation_enabled"] and int(self.config["saturation"]) != 0:
                            frame = ImageEnhance.Color(frame) \
                                .enhance((int(self.config["saturation"]) / 100.) + 1.)

                        # Apply sharpness
                        if self.config["sharpness_enabled"] and int(self.config["sharpness"]) != 0:
                            frame = ImageEnhance.Sharpness(frame) \
                                .enhance((int(self.config["sharpness"]) / 100.) + 1.)

                        # Apply vignetting
                        if self.config["vignette_enabled"] and int(self.config["vignette"]) != 0:
                            vignette_strength_temp = int(self.config["vignette"]) / 100.
                            if vignette is None or vignette_strength != vignette_strength_temp:
                                logging.info("Initializing vignetting effect")
                                vignette_strength = vignette_strength_temp
                                width, height = frame.size
                                x, y = np.meshgrid(np.arange(width), np.arange(height))
                                center_x, center_y = width / 2, height / 2
                                a = 1
                                b = 1
                                if width > height:
                                    b = height / width
                                elif height > width:
                                    a = width / height
                                distance = np.sqrt(((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2)
                                max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
                                normalized_distance = distance / max_distance
                                vignette = 1.0 - vignette_strength * (normalized_distance ** 2)
                                vignette = np.clip(vignette, 0, 1)
                                vignette = np.reshape(vignette, vignette.shape + (1,))
                            frame = Image.fromarray((vignette * np.asarray(frame, dtype=np.uint8)).astype(np.uint8))

                        # Apply bloom
                        if self.config["bloom_enabled"] and int(self.config["bloom"]) != 0:
                            if not bloom_initialized:
                                logging.info("Initializing bloom effect")
                                os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
                                import pygame
                                from bloom.bloom import bloom_effect24
                                bloom_initialized = True
                            pygame_surface = pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode)
                            pygame_surface = bloom_effect24(pygame_surface, 255 - int(self.config["bloom"]))
                            surface_data = pygame.surfarray.array3d(pygame_surface).swapaxes(0, 1)
                            frame = Image.fromarray(np.asarray(surface_data, dtype=np.uint8))

                        # Initialize file and write processed frame to the file
                        if render_to_file:
                            # First run -> start writing to the file
                            if not file_started:
                                # Main command
                                command = [self.ffmpeg_path.value,
                                           "-y",
                                           "-f", "rawvideo",
                                           "-vcodec", "rawvideo",
                                           "-s", f"{frame.size[0]}x{frame.size[1]}",
                                           "-pix_fmt", "rgb24",
                                           "-r", str(video_parameters["fps"]) if video_parameters["fps"] > 0 else "30",
                                           "-i", "-",
                                           "-an", "-vcodec", codec]

                                # Video parameters
                                if format_config == 0 or format_config == 1:
                                    # Output file pixel format
                                    command.append("-pix_fmt")
                                    command.append("yuv420p")

                                    # Video bitrate
                                    if bitrate > 0:
                                        command.append("-b:v")
                                        command.append(str(bitrate))

                                # MJPEG parameters
                                if format_config == 2:
                                    # Set highest MJPEG quality
                                    command.append("-q:v")
                                    command.append("1")

                                # Output file_s
                                command.append(render_temp_file)

                                # Start process
                                logging.info("Running {}".format(" ".join(command)))
                                ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                                                  stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

                                # Set started flag
                                file_started = True

                            # Write processed frame to file
                            if ffmpeg_process is not None:
                                ffmpeg_process.stdin.write(frame.tobytes("raw"))

                        # Limit fps if rendering to the preview
                        if not render_to_file:
                            while time.time() - time_started < 1. \
                                    / (30 if video_parameters["fps"] == 0 else video_parameters["fps"]):
                                time.sleep(.001)

                        # Calculate fps
                        fps = 0
                        if time_started > 0:
                            fps = 1. / (time.time() - time_started)

                        # Save time for next cycle
                        time_started = time.time()

                        # Filter FPS
                        if fps_filtered == 0:
                            fps_filtered = fps
                        else:
                            fps_filtered = PREVIEW_FPS_FILTER_K * fps_filtered + (1. - PREVIEW_FPS_FILTER_K) * fps

                        # Set playback info
                        time_current = frames_to_time_str(max(frame_to - (frame_from + current_frame - 1), 0),
                                                          frame_to, video_parameters["fps"])
                        time_total = frames_to_time_str(0, frame_to, video_parameters["fps"])
                        percentage_absolute = ((frame_from + current_frame - 1) / frame_to) * 100.
                        percentage_relative = (current_frame / (frame_to - frame_from + 1)) * 100.
                        self.playback_info.value = "Frame {}/{} ({}/{})" \
                                                   " ({:04.1f}%) Rendering FPS: {}".format(frame_from +
                                                                                           current_frame - 1,
                                                                                           frame_to,
                                                                                           time_current,
                                                                                           time_total,
                                                                                           round(percentage_absolute,
                                                                                                 1),
                                                                                           round(fps_filtered, 2))
                        self.rendering_progress.value = int(percentage_relative)
                        self.current_frame_absolute.value = frame_from + current_frame - 1

                        # Put to preview
                        if self.config["preview_mode"] == 0:
                            self.preview_queue.put(Image.open(os.path.join(frames_dir_input, filename)).convert("RGB"))
                        else:
                            self.preview_queue.put(frame)

                        # Delete both input and output frames
                        with self._file_scan_lock:
                            if os.path.exists(os.path.join(frames_dir_input, filename)):
                                os.remove(os.path.join(frames_dir_input, filename))
                            os.remove(os.path.join(frames_dir_output, filename))
                            self._extracted_frames.remove(current_frame)
                            self._processed_frames.remove(current_frame)

                        # Increment frame for next cycle
                        current_frame += 1

                        # Stop rendering if we passed current_frame += 1 on last frame
                        rendering_finished = current_frame > frame_to - frame_from + 1

                        # Pause or next frame request
                        if self.rendering_pause_request.value or self.rendering_single_frame_forward_request.value:
                            # Pause ntsc process
                            if not self.rendering_single_frame_forward_request.value:
                                logging.info("Pausing rendering process")
                                self._ntsc_processor_pause()

                            # Clear pause flag and set paused flag
                            self.rendering_pause_request.value = False
                            self.rendering_process_paused.value = True

                            # Wait for resume request or single frame request and stop request
                            while not self.rendering_resume_request.value \
                                    and not self.rendering_single_frame_forward_request.value \
                                    and not self.rendering_stop_request.value:
                                time.sleep(0.01)

                            if not self.rendering_stop_request.value:
                                # Pause on next cycle and clear single frame request flag
                                if self.rendering_single_frame_forward_request.value:
                                    self.rendering_pause_request.value = True
                                    self.rendering_single_frame_forward_request.value = False

                                # Resume process
                                logging.info("Resuming rendering process")
                                self._ntsc_processor_resume()

                                # Clear resume request and paused flag
                                self.rendering_resume_request.value = False
                                self.rendering_process_paused.value = False

                    except Exception as e:
                        logging.warning("Unable to read and render {} frame!".format(current_frame), exc_info=e)

                # Stop rendering if we passed current_frame += 1 on last frame
                if not rendering_finished:
                    # Load new batch
                    extracted_frames_num = 1 if len(self._extracted_frames) == 0 else max(
                        self._extracted_frames)
                    if extracted_frames_num <= current_frame + BUFFER_SIZE_FRAMES // 2 \
                            and extracted_frames_num < frame_to - frame_from + 1:
                        extract_frames_from = frame_from + extracted_frames_num
                        extract_frames_to = min(extract_frames_from + BUFFER_SIZE_FRAMES, frame_to)

                        # Extract remaining frames regarding of buffer size
                        if extract_frames_to + BUFFER_SIZE_FRAMES > frame_to:
                            extract_frames_to = frame_to

                        # Pause NTSC process before extracting frames (to prevent out of sync)
                        if not ntsc_paused_too_fast:
                            self._ntsc_processor_pause()

                        # Extract frames
                        self.extract_frames(file, extract_frames_from, extract_frames_to,
                                            name_from=extracted_frames_num + 1)

                        # Resume NTSC process after extracting frames
                        if not ntsc_paused_too_fast:
                            self._ntsc_processor_resume()

                    # Prevent out of sync ntsc process
                    if len(self._processed_frames) > 0:
                        if max(self._processed_frames) >= current_frame + BUFFER_SIZE_FRAMES // 2:
                            if not ntsc_paused_too_fast:
                                logging.warning("NTSC process is too fast or rendering process is too slow!")
                                self._ntsc_processor_pause()
                                ntsc_paused_too_fast = True
                        elif max(self._processed_frames) <= current_frame + 1:
                            if ntsc_paused_too_fast:
                                self._ntsc_processor_resume()
                                ntsc_paused_too_fast = False
                    else:
                        if ntsc_paused_too_fast:
                            self._ntsc_processor_resume()
                            ntsc_paused_too_fast = False

                # Stop request or rendering finished
                if self.rendering_stop_request.value or rendering_finished:
                    logging.info(("Finishing" if rendering_finished else "Stopping") + " rendering process")
                    self._ntsc_processor_stop()
                    try:
                        if ffmpeg_process is not None:
                            logging.info("Waiting for ffmpeg process to finish")
                            ffmpeg_process.stdin.close()
                            ffmpeg_process.wait()
                    except Exception as e:
                        logging.warning("Error stopping ffmpeg process!", exc_info=e)

                    # Save as
                    if render_to_file:
                        try:
                            # Move as video file
                            if format_config == 0 or format_config == 1:
                                logging.info("Moving rendered file to {}".format(render_to_file))
                                shutil.move(render_temp_file, render_to_file)

                            # Copy as directory with frames
                            if format_config == 2:
                                logging.info("Copying rendered frames to {}".format(render_to_file))
                                shutil.copytree(os.path.join("temp", "render"), render_to_file,
                                                ignore_dangling_symlinks=True, dirs_exist_ok=True)
                        except Exception as e:
                            logging.error("Error saving file!", exc_info=e)

                    if os.path.exists("temp"):
                        logging.info("Deleting temp directory")
                        shutil.rmtree("temp")
                    break

        except Exception as e:
            logging.error("Rendering error!", exc_info=e)

        # Reset flags and variables
        self.rendering_pause_request.value = False
        self.rendering_resume_request.value = False
        self.rendering_single_frame_forward_request.value = False
        self.rendering_stop_request.value = False
        self.rendering_process_active.value = False
        self.rendering_process_paused.value = False
        self.rendering_progress.value = 100

        # Callback
        self.preview_queue.put(1 if rendering_finished else 0)

        # Done
        logging.info("Rendering process finished")

    def _ntsc_processor_start(self, frames_num: int, video_parameters: dict) -> None:
        """
        Starts frame rendering (NTSC) process in thread
        :param frames_num:
        :param video_parameters:
        :return:
        """
        if self._ntsc_process is not None:
            return
        # Start thread
        ntsc_processor_thread = threading.Thread(target=self._ntsc_processor_thread_loop,
                                                 args=(frames_num, video_parameters,))
        ntsc_processor_thread.start()
        logging.info("Frame processor thread: {}".format(ntsc_processor_thread.name))

    def _ntsc_processor_pause(self) -> None:
        """
        Suspends frame rendering process
        :return:
        """
        if self._ntsc_process is not None:
            logging.info("Pausing NTSC process")
            psutil.Process(pid=self._ntsc_process.pid).suspend()

    def _ntsc_processor_resume(self) -> None:
        """
        Resumes frame rendering process
        :return:
        """
        if self._ntsc_process is not None:
            logging.info("Resuming NTSC process")
            psutil.Process(pid=self._ntsc_process.pid).resume()

    def _ntsc_processor_stop(self) -> None:
        """
        Stops frame rendering process
        :return:
        """
        if self._ntsc_process is not None:
            try:
                logging.info("Killing NTSC process")
                self._ntsc_process.kill()
                try:
                    self._ntsc_process.wait()
                except:
                    pass
                self._ntsc_process = None
            except Exception as e:
                logging.error("Error killing NTSC process!", exc_info=e)

    def _ntsc_processor_thread_loop(self, frames_num: int, video_parameters: dict) -> None:
        """
        Starts frame rendering process and monitors rendered frames
        :param frames_num:
        :param video_parameters:
        :return:
        """
        # Generate folders if not exist
        frames_dir = os.path.join("temp", "frames")
        if not os.path.exists(frames_dir):
            logging.info("Creating frames directory")
            os.makedirs(frames_dir)
        frames_dir = os.path.join("temp", "output")
        if not os.path.exists(frames_dir):
            logging.info("Creating output directory")
            os.makedirs(frames_dir)

        # Copy executables
        logging.info("Copying executables")
        ntsc_video_executable = os.path.join("temp",
                                             "ntsc_video" + os.path.splitext(self.ntsc_video_path.value)[1])
        ntsc_video_vhs_executable = os.path.join("temp",
                                                 "ntsc_vhs_video" + os.path.splitext(self.ntsc_video_vhs_path.value)[1])
        shutil.copyfile(self.ntsc_video_path.value, ntsc_video_executable)
        shutil.copyfile(self.ntsc_video_vhs_path.value, ntsc_video_vhs_executable)

        # Generate command
        ntsc_vhs_arguments = "-o"
        if self.config["monochrome"]:
            ntsc_vhs_arguments += "m"
        if self.config["scan_mode"] == 0:
            ntsc_vhs_arguments += "p"
        if self.config["fill_between_scan_lines"]:
            ntsc_vhs_arguments += "s"
        if self.config["system"] == 1 and self.config["mess_bottom_line"]:
            ntsc_vhs_arguments += "a"
        frame_width = int(video_parameters["width"] if self.config["out_size_equal_to_input"]
                          else self.config["out_width"])
        frame_height = int(video_parameters["height"] if self.config["out_size_equal_to_input"]
                           else self.config["out_height"])
        command = [ntsc_video_executable if self.config["system"] == 0 else ntsc_video_vhs_executable,
                   ntsc_vhs_arguments,
                   str(frames_num + 1),
                   str(frame_width),
                   str(frame_height),
                   str(int(self.config["noise"]))]

        # Start process
        logging.info("Running {}".format(" ".join(command)))
        self._ntsc_process = subprocess.Popen(command,
                                              stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL,
                                              cwd="temp")

        # Monitor files while process is running
        frames_dir = os.path.join("temp", "output")
        while self._ntsc_process is not None and self._ntsc_process.poll() is None:
            with self._file_scan_lock:
                # Get available frames
                processed_frames_temp = []
                for file in os.listdir(frames_dir):
                    if os.path.splitext(file)[1].lower() == ".bmp":
                        try:
                            frame_number = int(os.path.splitext(file)[0].strip())
                            processed_frames_temp.append(frame_number)
                        except:
                            pass

                # Update global list
                for processed_frame in processed_frames_temp:
                    if processed_frame not in self._processed_frames:
                        self._processed_frames.append(processed_frame)
                for processed_frame_old in self._processed_frames:
                    if processed_frame_old not in processed_frames_temp:
                        self._processed_frames.remove(processed_frame_old)

        # Process will be finished here (logging not working here and idk why)
        if self._ntsc_process is not None and self._ntsc_process.stdout is not None:
            self._ntsc_process.stdout.close()
            self._ntsc_process.wait()
        self._ntsc_process = None

    def extract_frames(self, file: str, frames_from: int, frames_to: int, name_from=1) -> None:
        """
        Extracts frames from file using multiple processes using ffmpeg
        :param file: file to extract frames from
        :param frames_from: 1 - N
        :param frames_to: 1 - N
        :param name_from: name of the first frame
        :return:
        """
        # Create frames directory if not exists
        frames_dir = os.path.join("temp", "frames")
        if not os.path.exists(frames_dir):
            logging.info("Creating temp directories")
            os.makedirs(frames_dir)

        # Log
        logging.info("Extracting {} frames from {} to {}. Output files: {} to {}".format(frames_to - frames_from + 1,
                                                                                         frames_from,
                                                                                         frames_to,
                                                                                         name_from,
                                                                                         name_from +
                                                                                         frames_to - name_from))

        # Extract frames
        command_ = [self.ffmpeg_path.value,
                    "-i", file,
                    "-vf", "select=\'between(n\\,{0}\\,{1})\'".format(frames_from - 1, frames_to - 1),
                    "-start_number", str(name_from),
                    "-vsync", "0",
                    os.path.join(frames_dir, "%06d.bmp")]
        _run_command_and_capture_output(command_)

        # Add them to list
        for i_ in range(name_from, name_from + (frames_to - frames_from + 1)):
            self._extracted_frames.append(i_)

    def count_frames(self, file: str) -> int:
        """
        Counts number of frames in file using ffprobe
        :param file: path to file
        :return: number of frames
        """
        try:
            command = [self.ffprobe_path.value,
                       "-v", "error",
                       "-select_streams", "v:0",
                       "-count_packets",
                       "-show_entries",
                       "stream=nb_read_packets",
                       "-of", "csv=p=0",
                       file]
            try:
                return int(_run_command_and_capture_output(command).replace("\r", "").replace("\n", "").strip())
            except:
                return 0

        except Exception as e:
            logging.error("Error retrieving number of frames in {}".format(file), exc_info=e)
        return 0

    def get_video_parameters(self, file: str) -> (dict | None):
        """
        Retrieves info about video file using ffprobe
        :param file: path to file
        :return: {"fps": ..., "width": ..., "height": ..., "codec": ..., "bit_rate": ...} or None if video is corrupted
        """
        try:
            command = [self.ffprobe_path.value,
                       "-v", "0",
                       "-of", "csv=p=0",
                       "-select_streams", "v:0",
                       "-show_entries", "stream=r_frame_rate,width,height,codec_name,bit_rate", file]
            out = _run_command_and_capture_output(command)
            if "/" in out and len(out.split(",")) == 5:
                out_parts = out.split(",")
                bit_rate = 0
                try:
                    bit_rate = int(out_parts[4].strip())
                except Exception as e:
                    logging.warning("Cannot determine file bit rate! {}".format(str(e)))

                framerate = float(out_parts[3].split("/")[0]) / float(out_parts[3].split("/")[1])
                return {
                    "fps": framerate,
                    "width": int(out_parts[1]),
                    "height": int(out_parts[2]),
                    "codec": out_parts[0].strip(),
                    "bit_rate": bit_rate
                }
        except Exception as e:
            logging.error("Error retrieving video info from {}".format(file), exc_info=e)
        return None
