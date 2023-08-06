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

import ctypes
import logging
import multiprocessing
import os
import sys
import threading
import time
import webbrowser

from PyQt5 import uic, QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QProgressDialog

from FramesProcessor import FramesProcessor
from JSONReaderWriter import save_json

# GUI stylesheet
STYLESHEET_FILE = "stylesheet.qss"

# Video and image formats supported by ffmpeg (I hope so)
MEDIA_FILES = ["*.avi", "*.mkv", "*.mpg", "*.mpeg", "*.wmv", "*.flv", "*.webm", "*.m4v", "*.3gp", "*.mp4", "*.mov",
               "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tiff", "*.tif", "*.webp", "*.exr", "*.ppm",
               "*.pgm", "*.pbm", "*.pnm", "*.raw", "*.svg"]


class GUI:
    def __init__(self, config: dict, config_file: str, version: str, frames_processor: FramesProcessor,
                 statusbar_queue: multiprocessing.Queue) -> None:
        # Replace icon in taskbar
        if os.name == "nt":
            logging.info("Replacing icon in taskbar")
            app_ip = "f3rni.ntscvhsvideorenderer.ntscvhsvideorenderer." + version
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_ip)

        # Start app
        logging.info("Opening GUI")
        app = QApplication.instance() or QApplication(sys.argv)
        app.setStyle("windows")
        win = Window(config, config_file, version, frames_processor, statusbar_queue)
        app.exec_()
        logging.info("GUI closed")


class Window(QMainWindow):
    statusbar_show_message_signal = QtCore.pyqtSignal(str)  # QtCore.Signal(str)
    lb_preview_set_pixmap_signal = QtCore.pyqtSignal(QPixmap)  # QtCore.Signal(QPixmap)
    rendering_done_signal = QtCore.pyqtSignal(int)  # QtCore.Signal(int)
    rendering_progress_set_value_signal = QtCore.pyqtSignal(int)  # QtCore.Signal(int)
    set_timeline_frame_signal = QtCore.pyqtSignal()  # QtCore.Signal()

    def __init__(self, config: dict, config_file: str, version: str, frames_processor: FramesProcessor,
                 statusbar_queue: multiprocessing.Queue) -> None:
        super(Window, self).__init__()

        self.config = config
        self.config_file = config_file
        self.version = version
        self.frames_processor = frames_processor
        self.statusbar_queue = statusbar_queue

        self.current_file = None
        self.render_to_file = None
        self.rendering_progress = None

        # Load GUI from file
        uic.loadUi("gui.ui", self)

        with open(STYLESHEET_FILE, "r") as stylesheet_file:
            self.setStyleSheet(stylesheet_file.read())

        # Set window title
        self.setWindowTitle("NTSC-VHS-Renderer " + version)

        # Set icon
        self.setWindowIcon(QtGui.QIcon("icon.png"))

        # Show GUI
        self.show()
        self.setAcceptDrops(True)

        # Connect signals
        self.statusbar_show_message_signal.connect(lambda message: self.statusbar.showMessage(message))
        self.lb_preview_set_pixmap_signal.connect(self.lb_preview.setPixmap)
        self.rendering_done_signal.connect(self.rendering_done)
        self.set_timeline_frame_signal.connect(self.set_timeline_frame)

        # Connect menu buttons
        self.actionOpen.triggered.connect(self.open_file)
        self.actionAbout_2.triggered.connect(self.menu_about)
        self.actionGitHub_page.triggered.connect(lambda _:
                                                 webbrowser.open("https://github.com/F33RNI/NTSC-VHS-Renderer",
                                                                 new=0, autoraise=True))

        # Connect buttons
        self.btn_start_render.clicked.connect(self.start_file_rendering)
        self.btn_next.clicked.connect(self.control_next_frame)
        self.btn_previous.clicked.connect(self.control_previous_frame)
        self.btn_play_pause.clicked.connect(self.control_play_pause)
        self.sl_timeline.sliderReleased.connect(self.timeline_callback)
        self.sl_timeline.sliderPressed.connect(self.pause_rendering)

        # Set gui elements from config
        self.rb_preview_original.setChecked(self.config["preview_mode"] == 0)
        self.rb_preview_processed.setChecked(self.config["preview_mode"] == 1)
        self.rb_system_ntsc.setChecked(self.config["system"] == 0)
        self.rb_system_vhs.setChecked(self.config["system"] == 1)
        self.cb_monochrome.setChecked(self.config["monochrome"])
        self.sb_noise.setValue(int(self.config["noise"]))
        self.rb_progressive.setChecked(self.config["scan_mode"] == 0)
        self.rb_interlaced.setChecked(self.config["scan_mode"] == 1)
        self.cb_fill_between_scan_lines.setChecked(self.config["fill_between_scan_lines"])
        self.cb_mess_bottom_line.setChecked(self.config["mess_bottom_line"])
        self.sb_width.setValue(int(self.config["out_width"]))
        self.sb_height.setValue(int(self.config["out_height"]))
        self.cb_size_input.setChecked(self.config["out_size_equal_to_input"])
        self.cb_brightness_enabled.setChecked(self.config["brightness_enabled"])
        self.sb_brightness.setValue(int(self.config["brightness"]))
        self.cb_contrast_enabled.setChecked(self.config["contrast_enabled"])
        self.sb_contrast.setValue(int(self.config["contrast"]))
        self.cb_sharpness_enabled.setChecked(self.config["sharpness_enabled"])
        self.sb_sharpness.setValue(int(self.config["sharpness"]))
        self.cb_saturation_enabled.setChecked(self.config["saturation_enabled"])
        self.sb_saturation.setValue(int(self.config["saturation"]))
        self.cb_bloom_enabled.setChecked(self.config["bloom_enabled"])
        self.sb_bloom.setValue(int(self.config["bloom"]))
        self.sb_frame_from.setValue(int(self.config["render_frame_from"]))
        self.cb_from_start.setChecked(self.config["render_frame_from_start"])
        self.sb_frame_to.setValue(int(self.config["render_frame_to"]))
        self.cb_to_end.setChecked(self.config["render_frame_to_end"])
        self.rb_format_mp4.setChecked(self.config["out_format"] == 0)
        self.rb_format_mov.setChecked(self.config["out_format"] == 1)
        self.rb_format_jpg.setChecked(self.config["out_format"] == 2)
        self.sb_bitrate.setValue(int(self.config["out_bitrate"]))
        self.cb_bitrate_same.setChecked(self.config["out_bitrate_equal_to_input"])

        # Connect config updater
        self.rb_preview_original.clicked.connect(lambda _: self.update_config(False))
        self.rb_preview_processed.clicked.connect(lambda _: self.update_config(False))

        self.rb_system_ntsc.clicked.connect(lambda _: self.update_config(True))
        self.rb_system_vhs.clicked.connect(lambda _: self.update_config(True))
        self.cb_monochrome.clicked.connect(lambda _: self.update_config(True))
        self.sb_noise.valueChanged.connect(lambda _: self.update_config(True))
        self.rb_progressive.clicked.connect(lambda _: self.update_config(True))
        self.rb_interlaced.clicked.connect(lambda _: self.update_config(True))
        self.cb_fill_between_scan_lines.clicked.connect(lambda _: self.update_config(True))
        self.cb_mess_bottom_line.clicked.connect(lambda _: self.update_config(True))
        self.sb_width.valueChanged.connect(lambda _: self.update_config(True))
        self.sb_height.valueChanged.connect(lambda _: self.update_config(True))
        self.cb_size_input.clicked.connect(lambda _: self.update_config(True))

        self.cb_brightness_enabled.clicked.connect(lambda _: self.update_config(False))
        self.sb_brightness.valueChanged.connect(lambda _: self.update_config(False))
        self.cb_contrast_enabled.clicked.connect(lambda _: self.update_config(False))
        self.sb_contrast.valueChanged.connect(lambda _: self.update_config(False))
        self.cb_sharpness_enabled.clicked.connect(lambda _: self.update_config(False))
        self.sb_sharpness.valueChanged.connect(lambda _: self.update_config(False))
        self.cb_saturation_enabled.clicked.connect(lambda _: self.update_config(False))
        self.sb_saturation.valueChanged.connect(lambda _: self.update_config(False))
        self.cb_bloom_enabled.clicked.connect(lambda _: self.update_config(False))
        self.sb_bloom.valueChanged.connect(lambda _: self.update_config(False))

        self.sb_frame_from.valueChanged.connect(lambda _: self.update_config(True))
        self.cb_from_start.clicked.connect(lambda _: self.update_config(True))
        self.sb_frame_to.valueChanged.connect(lambda _: self.update_config(True))
        self.cb_to_end.clicked.connect(lambda _: self.update_config(True))
        self.rb_format_mp4.clicked.connect(lambda _: self.update_config(True))
        self.rb_format_mov.clicked.connect(lambda _: self.update_config(True))
        self.rb_format_jpg.clicked.connect(lambda _: self.update_config(True))
        self.sb_bitrate.valueChanged.connect(lambda _: self.update_config(True))
        self.cb_bitrate_same.clicked.connect(lambda _: self.update_config(True))

        # Connect resize listener
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.installEventFilter(self)

        # Start status bar updater
        self.start_statusbar_handler()

        # Update config file
        self.update_config(False)

        # Start preview updater
        self.start_preview_updater()

        # Done
        logging.info("GUI loading finished")

    def update_config(self, stop_rendering: bool) -> None:
        """
        Saves gui fields to config file
        :param stop_rendering: Do we need to stop rendering before updating config file?
        :return:
        """
        # Stop rendering
        if stop_rendering:
            self.stop_rendering()

        # Read data from elements
        self.config["preview_mode"] = 0 if self.rb_preview_original.isChecked() else 1
        self.config["system"] = 0 if self.rb_system_ntsc.isChecked() else 1
        self.config["monochrome"] = self.cb_monochrome.isChecked()
        self.config["noise"] = int(self.sb_noise.value())
        self.config["scan_mode"] = 0 if self.rb_progressive.isChecked() else 1
        self.config["fill_between_scan_lines"] = self.cb_fill_between_scan_lines.isChecked()
        self.config["mess_bottom_line"] = self.cb_mess_bottom_line.isChecked()
        self.config["out_width"] = int(self.sb_width.value())
        self.config["out_height"] = int(self.sb_height.value())
        self.config["out_size_equal_to_input"] = self.cb_size_input.isChecked()
        self.config["brightness_enabled"] = self.cb_brightness_enabled.isChecked()
        self.config["brightness"] = int(self.sb_brightness.value())
        self.config["contrast_enabled"] = self.cb_contrast_enabled.isChecked()
        self.config["contrast"] = int(self.sb_contrast.value())
        self.config["sharpness_enabled"] = self.cb_sharpness_enabled.isChecked()
        self.config["sharpness"] = int(self.sb_sharpness.value())
        self.config["saturation_enabled"] = self.cb_saturation_enabled.isChecked()
        self.config["saturation"] = int(self.sb_saturation.value())
        self.config["bloom_enabled"] = self.cb_bloom_enabled.isChecked()
        self.config["bloom"] = int(self.sb_bloom.value())
        self.config["render_frame_from"] = int(self.sb_frame_from.value())
        self.config["render_frame_from_start"] = self.cb_from_start.isChecked()
        self.config["render_frame_to"] = int(self.sb_frame_to.value())
        self.config["render_frame_to_end"] = self.cb_to_end.isChecked()
        self.config["out_format"] = 0 if self.rb_format_mp4.isChecked() \
            else (1 if self.rb_format_mov.isChecked()
                  else 2)
        self.config["out_bitrate"] = int(self.sb_bitrate.value())
        self.config["out_bitrate_equal_to_input"] = self.cb_bitrate_same.isChecked()

        # Enable / disable some elements
        self.cb_mess_bottom_line.setEnabled(self.rb_system_vhs.isChecked())
        self.sb_width.setEnabled(not self.cb_size_input.isChecked())
        self.sb_height.setEnabled(not self.cb_size_input.isChecked())
        self.sb_brightness.setEnabled(self.cb_brightness_enabled.isChecked())
        self.sb_contrast.setEnabled(self.cb_contrast_enabled.isChecked())
        self.sb_sharpness.setEnabled(self.cb_sharpness_enabled.isChecked())
        self.sb_saturation.setEnabled(self.cb_saturation_enabled.isChecked())
        self.sb_bloom.setEnabled(self.cb_bloom_enabled.isChecked())
        self.sb_frame_from.setEnabled(not self.cb_from_start.isChecked())
        self.sb_frame_to.setEnabled(not self.cb_to_end.isChecked())
        self.sb_bitrate.setEnabled(not self.cb_bitrate_same.isChecked())

        # Save to file
        save_json(self.config_file, self.config)

    def control_next_frame(self) -> None:
        """
        Next frame button callback
        :return:
        """
        # Rendering is active
        if self.frames_processor.rendering_process_active.value:
            # No requests yet
            if not self.frames_processor.rendering_pause_request.value \
                    and not self.frames_processor.rendering_resume_request.value:
                # Pause rendering and request next frame
                self.frames_processor.rendering_pause_request.value = True
                self.frames_processor.rendering_single_frame_forward_request.value = True
                while self.frames_processor.rendering_pause_request.value \
                        or self.frames_processor.rendering_single_frame_forward_request.value:
                    time.sleep(0.01)
                self.btn_play_pause.setText("Play")

        # Play file again and pause it
        elif self.current_file:
            self.open_file(self.current_file, render_to_file=False)
            self.pause_rendering()

    def control_previous_frame(self) -> None:
        """
        Previous frame button callback
        :return:
        """
        # Rendering is active
        if self.frames_processor.rendering_process_active.value:
            # No requests yet
            if not self.frames_processor.rendering_pause_request.value \
                    and not self.frames_processor.rendering_resume_request.value:
                # Stop current rendering
                self.stop_rendering()

        # Play file again from previous frame and pause it
        if self.current_file:
            self.open_file(self.current_file, render_to_file=False,
                           render_from_frame=max(min(max(self.frames_processor.current_frame_absolute.value - 1, 1),
                                                 self.frames_processor.frames_max.value - 1), 1))
            self.pause_rendering()

    def control_play_pause(self) -> None:
        """
        Play / Pause button control
        :return:
        """
        # Rendering is active
        if self.frames_processor.rendering_process_active.value:
            # No requests yet
            if not self.frames_processor.rendering_pause_request.value \
                    and not self.frames_processor.rendering_resume_request.value:
                # Pause rendering
                if self.btn_play_pause.text() == "Pause":
                    self.pause_rendering()

                # Resume rendering
                else:
                    self.resume_rendering()

        # Play file again
        elif self.current_file:
            self.open_file(self.current_file,
                           render_to_file=False,
                           render_from_frame=self.frames_processor.current_frame_absolute.value)

    def timeline_callback(self) -> None:
        """
        Timeline value changed callback
        :return:
        """
        # Store current setting
        paused_previously = self.btn_play_pause.text() == "Play"

        # Rendering is active
        if self.frames_processor.rendering_process_active.value:
            # No requests yet
            if not self.frames_processor.rendering_pause_request.value \
                    and not self.frames_processor.rendering_resume_request.value:
                # Stop current rendering
                self.stop_rendering()

        # Play file again from selected frame
        if self.current_file:
            self.open_file(self.current_file, render_to_file=False,
                           render_from_frame=min(max(self.sl_timeline.value(), 1),
                                                 self.frames_processor.frames_max.value - 1))
            # Previously paused -> pause now too
            if paused_previously:
                self.pause_rendering()

    def start_preview_updater(self) -> None:
        """
        Updates preview image label
        :return:
        """

        def _queue_reader_thread() -> None:
            while True:
                preview_image = self.frames_processor.preview_queue.get()

                # None -> Exit
                if preview_image is None:
                    break

                # 0 -> Rendering finished unsuccessfully / aborted. 1 -> Rendering finished successfully
                if preview_image == 0 or preview_image == 1:
                    self.rendering_done_signal.emit(int(preview_image))
                    continue

                try:
                    # Convert to pixmap
                    data = preview_image.tobytes()
                    qim = QImage(data, preview_image.size[0], preview_image.size[1], 3 * preview_image.size[0],
                                 QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qim)

                    # Set pixmap
                    self.lb_preview_set_pixmap_signal.emit(pixmap.scaled(self.lb_preview.width(),
                                                                         self.lb_preview.height(),
                                                                         Qt.KeepAspectRatio))

                    # Set video info
                    self.lb_progress.setText(self.frames_processor.playback_info.value)

                    # Set timeline progress
                    self.set_timeline_frame_signal.emit()

                    # Enable preview selectors
                    self.rb_preview_original.setEnabled(True)
                    self.rb_preview_processed.setEnabled(True)

                    # Set QProgressDialog progress
                    try:
                        if self.render_to_file and self.rendering_progress is not None:
                            self.rendering_progress_set_value_signal \
                                .emit(int(self.frames_processor.rendering_progress.value))
                    except Exception as e:
                        logging.warning("Cannot update QProgressDialog! {}".format(str(e)))
                except Exception as e:
                    logging.error("Error updating preview image!", exc_info=e)

        preview_updater_thread = threading.Thread(target=_queue_reader_thread)
        preview_updater_thread.start()
        logging.info("Preview updater thread: {}".format(preview_updater_thread.name))

    def set_timeline_frame(self) -> None:
        """
        Sets current frame on timeline using self.frames_processor.frames_max
        and self.frames_processor.current_frame_absolute.value
        :return:
        """
        self.sl_timeline.blockSignals(True)
        self.sl_timeline.setMaximum(self.frames_processor.frames_max.value)
        self.sl_timeline.setValue(self.frames_processor.current_frame_absolute.value)
        self.sl_timeline.blockSignals(False)

    def rendering_done(self, successful: bool) -> None:
        """
        Callback for rendering finish
        :return:
        """
        # Ignore if process active again
        if self.frames_processor.rendering_process_active.value:
            return

        # Close QProgressDialog
        if self.rendering_progress is not None:
            self.rendering_progress.close()

        # Disable preview selectors
        self.rb_preview_original.setEnabled(False)
        self.rb_preview_processed.setEnabled(False)

        # Change button text to Play
        self.btn_play_pause.setText("Play")

        # Enable controls back
        if self.render_to_file:
            self.btn_previous.setEnabled(True)
            self.btn_play_pause.setEnabled(True)
            self.btn_next.setEnabled(True)
            self.sl_timeline.setEnabled(True)

        # Show successful dialog
        if successful and self.render_to_file:
            info_box = QMessageBox(self)
            info_box.setWindowTitle("Finished")
            info_box.setText("Rendering finished! File saved to:\n{}".format(str(self.render_to_file)))
            info_box.exec_()

    def start_statusbar_handler(self) -> None:
        """
        Prints logs to the status bar
        :return:
        """

        def _queue_reader_thread() -> None:
            while True:
                record = self.statusbar_queue.get()
                if record is None:
                    break
                try:
                    if self.statusbar_show_message_signal is not None:
                        self.statusbar_show_message_signal.emit(str(record.msg))
                except:
                    pass

                # Sleep some time to show all messages
                time.sleep(0.1)

        status_bar_thread = threading.Thread(target=_queue_reader_thread)
        status_bar_thread.start()
        logging.info("Status bar handling thread: {}".format(status_bar_thread.name))

    def menu_about(self) -> None:
        """
        Shows about message box
        :return:
        """
        about_box = QMessageBox(self)
        about_box.setWindowTitle("About")
        about_box.setText("NTSC-VHS-Renderer {}"
                          "\n\nCreated by Fern Lane (aka F3RNI)"
                          "\nNTSC-CRT library by LMP88959 (aka EMMIR)"
                          "\nFFmpeg library by www.ffmpeg.org"
                          "\nBloomEffect by Yoann Berenguer"
                          .format(self.version))
        about_box.exec_()

    def open_file(self, filename=None, render_to_file=None, render_from_frame=1) -> None:
        """
        Opens file using file dialog
        :param filename:
        :param render_to_file:
        :param render_from_frame:
        :return:
        """
        # Stop current rendering
        self.stop_rendering()

        # Build dialog
        if not filename or not os.path.exists(filename):
            options = QFileDialog.Options()
            options &= not QFileDialog.DontUseNativeDialog
            filename, _ = QFileDialog.getOpenFileName(self, "Open file", self.config["file_last"],
                                                      "Media Files ({});;All Files (*.*)".format(" ".join(MEDIA_FILES)),
                                                      options=options)
        if filename and os.path.exists(filename):
            # Update current file in config for next open
            self.config["file_last"] = filename
            self.update_config(True)

            # Save for future dialogs
            self.render_to_file = render_to_file

            # Check file
            video_parameters = self.frames_processor.get_video_parameters(filename)
            frames = self.frames_processor.count_frames(filename)
            logging.info("Found {} frames".format(frames))
            logging.info("Video parameters: {}".format(str(video_parameters)))
            if video_parameters and video_parameters["fps"] > 0 and frames > 0:
                if video_parameters["width"] > 0 and video_parameters["height"] > 0:
                    # If we are here, file seems OK
                    # Save current file
                    logging.info("Selected file: {}".format(filename))
                    self.current_file = filename

                    # Clear preview text
                    self.lb_preview.setText("")

                    # Show file info
                    self.lb_filename.setText("{0} ({1}, {2}x{3}, {4}FPS, {5}Kbps)"
                                             .format(os.path.normpath(filename),
                                                     video_parameters["codec"],
                                                     video_parameters["width"],
                                                     video_parameters["height"],
                                                     round(video_parameters["fps"], 2),
                                                     video_parameters["bit_rate"] // 1000))

                    render_to_frame = frames
                    if render_to_file:
                        # Get render frames range
                        if not self.config["render_frame_from_start"]:
                            render_from_frame = int(self.config["render_frame_from"])
                        if not self.config["render_frame_to_end"]:
                            render_to_frame = int(self.config["render_frame_to"])

                        # Show QProgressDialog if rendering to file
                        self.rendering_progress = QProgressDialog("Rendering file\n{}".format(filename), "",
                                                                  0, 100, self)
                        self.rendering_progress.setWindowTitle("Rendering")
                        self.rendering_progress.setCancelButtonText("Abort")
                        self.rendering_progress.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
                        self.rendering_progress.canceled.connect(self.abort_file_rendering)
                        self.rendering_progress.show()
                        try:
                            self.rendering_progress_set_value_signal.disconnect()
                        except:
                            pass
                        self.rendering_progress_set_value_signal.connect(self.rendering_progress.setValue)

                        # Go to the preview tab
                        self.tabWidget.setCurrentIndex(0)

                        # Set processed preview
                        self.rb_preview_original.setChecked(False)
                        self.rb_preview_processed.setChecked(True)
                        self.update_config(stop_rendering=False)

                    # Start rendering
                    logging.info("Start rendering from {} frame".format(render_from_frame))
                    self.frames_processor.start_rendering(filename, render_from_frame, render_to_frame,
                                                          video_parameters, render_to_file)

                    # Enable controls
                    self.btn_play_pause.setText("Pause")
                    self.btn_previous.setEnabled(not render_to_file)
                    self.btn_play_pause.setEnabled(not render_to_file)
                    self.btn_next.setEnabled(not render_to_file)
                    self.sl_timeline.setEnabled(not render_to_file)

                else:
                    self.error_message_box("Wrong dimensions of file {}".format(filename),
                                           "Please select another file")

            else:
                self.error_message_box("No frames found in file {}\n or file framerate is unknown".format(filename),
                                       "Please select another file")

    def start_file_rendering(self) -> None:
        """
        Asks path to save rendered file and starts file rendering
        :return:
        """
        # Check if we have loaded file
        if self.current_file and os.path.exists(self.current_file):
            # Stop current rendering
            self.stop_rendering()

            # Get render frames range
            frames = self.frames_processor.count_frames(self.current_file)
            render_from_frame = 1
            render_to_frame = frames
            if not self.config["render_frame_from_start"]:
                render_from_frame = int(self.config["render_frame_from"])
            if not self.config["render_frame_to_end"]:
                render_to_frame = int(self.config["render_frame_to"])

            # Check frames range
            if render_from_frame <= 0 or render_to_frame <= 0 \
                    or render_from_frame > render_to_frame or render_to_frame > frames:
                self.error_message_box("Wrong frame number", "Please make sure that:"
                                                             "\nRender to frame > Render from frame"
                                                             "\nRender to frame < Total number of frames")
                return

            # Use current file as prompt if no previous in config (without file extension)
            out_file_prompt = os.path.splitext(self.config["out_file_last"]
                                               if self.config["out_file_last"]
                                               else self.current_file)[0]

            # mp4 or mov file
            if self.config["out_format"] == 0 or self.config["out_format"] == 1:
                if self.config["out_format"] == 0:
                    format_filter = "MP4 File (*.mp4)"
                else:
                    format_filter = "MOV File (*.mov)"

                options = QFileDialog.Options()
                options &= not QFileDialog.DontUseNativeDialog
                filename, _ = QFileDialog.getSaveFileName(self, "Select where to save file",
                                                          out_file_prompt,
                                                          format_filter + ";;All Files (*.*)", options=options)

                # Add file extension if not specified
                if filename:
                    if self.config["out_format"] == 0 and not filename.lower().endswith(".mp4"):
                        filename += ".mp4"
                    if self.config["out_format"] == 1 and not filename.lower().endswith(".mov"):
                        filename += ".mov"

            # Directory with frames
            else:
                options = QFileDialog.Options()
                options &= not QFileDialog.DontUseNativeDialog
                filename = QFileDialog.getExistingDirectory(self, "Select where to save frames",
                                                            out_file_prompt,
                                                            options=options)

            # Check is user selected anything
            if filename:
                self.config["out_file_last"] = filename
                self.update_config(stop_rendering=False)
                self.open_file(self.current_file, render_to_file=filename)

        # No file
        else:
            self.error_message_box("Nothing to render!", "Please open existing file first")

    def abort_file_rendering(self) -> None:
        """
        Asks user to abort rendering to file (QProgressDialog cancel callback)
        :return:
        """
        # Stop updating QProgressDialog
        if self.rendering_progress is not None:
            self.rendering_progress.hide()

        # Prevent dialog after aborting
        if self.frames_processor.rendering_process_active.value:
            reply = QMessageBox.question(self, "Abort rendering", "Are you sure you want to stop rendering process?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                # Stop rendering =(
                self.stop_rendering()

            # Continue QProgressDialog
            else:
                if self.rendering_progress is not None:
                    self.rendering_progress.show()

    def pause_rendering(self) -> None:
        """
        Pauses rendering and waits until rendering is paused
        :return:
        """
        if self.frames_processor.rendering_process_active.value \
                and not self.frames_processor.rendering_process_paused.value:
            self.frames_processor.rendering_pause_request.value = True
            while self.frames_processor.rendering_pause_request.value:
                time.sleep(0.01)
            self.btn_play_pause.setText("Play")

    def resume_rendering(self) -> None:
        """
        Resumes rendering and waits until rendering is resumed
        :return:
        """
        if self.frames_processor.rendering_process_active.value \
                and self.frames_processor.rendering_process_paused.value:
            self.frames_processor.rendering_resume_request.value = True
            while self.frames_processor.rendering_resume_request.value:
                time.sleep(0.01)
            self.btn_play_pause.setText("Pause")

    def stop_rendering(self) -> None:
        """
        Stops rendering and waits until process is stopped
        :return:
        """
        if self.frames_processor.rendering_process_active.value:
            self.frames_processor.rendering_stop_request.value = True
            logging.info("Waiting for rendering process to finish")
            while self.frames_processor.rendering_process_active.value:
                time.sleep(0.01)

    def error_message_box(self, error_message: str, additional_text=None) -> None:
        """
        Shows error message box
        :param error_message:
        :param additional_text:
        :return:
        """
        error_box = QMessageBox(self)
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle("Error")
        error_box.setText(error_message)
        if additional_text:
            error_box.setInformativeText(str(additional_text))
        error_box.exec_()

    def dragEnterEvent(self, event):
        event.accept()

    def dragMoveEvent(self, event):
        source = event.source()
        target = self.childAt(event.pos())
        if target == self.lb_preview and target != source and not self.frames_processor.rendering_process_active.value:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """
        Loads file from drop
        :param event:
        :return:
        """
        source = event.source()
        target = self.childAt(event.pos())
        if target != self.lb_preview or target == source or self.frames_processor.rendering_process_active.value:
            return

        event.accept()

        try:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.open_file(file_path)
        except Exception as e:
            logging.error("Error loading dropped file!", exc_info=e)

    def mouseReleaseEvent(self, event):
        """
        Mouse click on preview label
        :param event:
        :return:
        """
        if self.childAt(event.pos()) == self.lb_preview:
            if self.current_file is None:
                self.open_file()

    def eventFilter(self, obj, event):
        """
        Automatically resizes preview label
        :param obj:
        :param event:
        :return:
        """
        if event.type() == QtCore.QEvent.Resize:
            if self.lb_preview is not None:
                if self.lb_preview.pixmap() is not None:
                    pixmap = self.lb_preview.pixmap().scaled(self.lb_preview.width(), self.lb_preview.height(),
                                                             Qt.KeepAspectRatio)
                    self.lb_preview.setPixmap(pixmap)
        return super().eventFilter(obj, event)

    def closeEvent(self, event) -> None:
        """
        Closes app (asks user before it if we have opened file)
        :param event:
        :return:
        """
        if self.current_file:
            reply = QMessageBox.question(self, "Quit", "Are you sure you want to quit?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                # Stop current rendering
                self.stop_rendering()

                # Stop preview updater thread
                self.frames_processor.preview_queue.put(None)

                # Accept event
                logging.info("Closing GUI")
                event.accept()
            else:
                event.ignore()

        # No file -> Exit without asking
        else:
            # Stop preview updater thread
            self.frames_processor.preview_queue.put(None)

            # Accept event
            logging.info("Closing GUI")
            event.accept()
