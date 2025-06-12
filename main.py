#                                      Program was created https://github.com/Lunsee
#
#
#
import os
from nudenet import NudeDetector
from PIL import Image, ImageDraw, ImageFilter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog, QLabel, QWidget, QComboBox, QLineEdit, QCheckBox,
    QMessageBox, QDialog, QListWidget, QProgressBar
)
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import subprocess
import ffmpeg
import datetime
import shutil
class CensorshipMemory:
    def __init__(self, memory_size=5):
        self.memory_size = memory_size
        self.history = []

    def update(self, detections):
        """update history"""
        self.history.append(detections)
        if len(self.history) > self.memory_size:
            self.history.pop(0)

    def get_active_detections(self):
        """return active detections"""
        for detections in reversed(self.history):
            if detections:
                return detections
        return []



model_path = os.path.join(os.path.dirname(__file__), "640m.onnx")
print (f"model path: {model_path}")

# classes list
censored_classes = ['FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED', 'BUTTOCKS_EXPOSED','FEMALE_BREAST_COVERED']
print (f"default censored_classes: {censored_classes}")
BASE_DIR = os.path.dirname(__file__)

def censor_image(image, output_path, detections,type_censor,mode,min_prob):
    print(f"Processing frame: {image}")
    print(f"Type censor: {type_censor}")
    image_np = np.array(image)


    # detections = detector.detect(image)
    print(f"Model response for image {image}: {detections}")
    if not detections:
        print(f"For image {image} no intimate places were found.")

    filtered_detections = [
        detection for detection in detections
        if detection['score'] >= min_prob
    ]
    #print(f"filtered_detections: {filtered_detections}")


    draw = ImageDraw.Draw(image)


    if type_censor == "Black rectangles":
        print(f"Тип цензуры: Black rectangles")
        try:
            if isinstance(filtered_detections, list):
                for detection in filtered_detections:
                    #print(f"Detection: {detection}")
                    if detection['class'] in censored_classes:
                        x1, y1, x2, y2 = detection['box']
                        draw.rectangle([x1, y1, x1 + x2, y1 + y2], fill="black")
            else:
                print(f"Invalid data structure for image {image}")
            if (mode == "video"):
                return image
            else:
                image.save(output_path)
                print(f"Image saved from: {output_path}")
        except Exception as e:
            print(f"Error with processing image {image}: {e}")
    elif type_censor == "Pixel":
        print(f"Тип цензуры: Pixel")
        try:

            if isinstance(filtered_detections, list):
                for detection in filtered_detections:
                    #print(f"Detection: {detection}")
                    if detection['class'] in censored_classes:
                        x1, y1, width, height = detection['box']

                        # lower right corner
                        x2 = x1 + width
                        y2 = y1 + height

                        # cut
                        cropped_image = image.crop(
                            (x1, y1, x2, y2))  # x1, y1 - upper left corner, x2, y2 - lower right corner


                        cropped_image_cv = np.array(cropped_image)
                        cropped_image_cv = cv2.cvtColor(cropped_image_cv, cv2.COLOR_RGB2BGR)

                        # PIXELS SIZE
                        w, h = (5, 5)

                        temp = cv2.resize(cropped_image_cv, (w, h), interpolation=cv2.INTER_LINEAR)
                        output_cens_item = cv2.resize(temp, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                        image.paste(Image.fromarray(cv2.cvtColor(output_cens_item, cv2.COLOR_BGR2RGB)), (x1, y1))
            else:
                print(f"Invalid data structure for image {image}")
            if(mode == "video"):
                return image
            else:
                image.save(output_path)
                print(f"Image saved from: {output_path}")
        except Exception as e:
            print(f"Error with processing image {image}: {e}")
    elif type_censor == "blur":
        print(f"Type censor: blur")
        #self.label_blur_size = QLabel("Blur power: (ONLY odd number, default - 55)")
        #self.select_blur_size = QLineEdit("50", parent=self)
        #size_blur = int(self.select_blur_size.text())
        #self.layout.addWidget(self.label_blur_size)
        #self.layout.addWidget(self.select_blur_size)

        try:
            if isinstance(filtered_detections, list):
                for detection in filtered_detections:
                    #print(f"Detection: {detection}")
                    if detection['class'] in censored_classes:
                        x1, y1, width, height = detection['box']

                        # lower right corner
                        x2 = x1 + width
                        y2 = y1 + height

                        # cut
                        cropped_image = image.crop(
                            (x1, y1, x2, y2))  # x1, y1 - upper left corner, x2, y2 - lower right corner

                        # numpy list
                        cropped_image_cv = np.array(cropped_image)
                        cropped_image_cv = cv2.cvtColor(cropped_image_cv, cv2.COLOR_RGB2BGR)

                        size_blur = 55

                        output_cens_item = cv2.GaussianBlur(cropped_image_cv,(size_blur, size_blur),0)
                        image.paste(Image.fromarray(cv2.cvtColor(output_cens_item, cv2.COLOR_BGR2RGB)), (x1, y1))
            else:
                print(f"Invalid data structure for image {image}")
            if(mode == "video"):
                return image
            else:
                image.save(output_path)
                print(f"Image saved from: {output_path}")
        except Exception as e:
            print(f"Error with processing image {image}: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.select_output_button = QPushButton("Select Output Directory")
        self.init_ui()
        self.detector = NudeDetector(model_path=model_path, inference_resolution=640)
        self.input_dir = None
        self.output_dir = None
        self.type_censor = "Pixel"
        self.min_prob = 0.1
        self.auto_create_dir_flag = False
        self.frame_skip = 1


    def init_ui(self):
        self.setWindowTitle("Censorship Tool")
        self.setGeometry(300, 300, 400, 200)

        layout = QVBoxLayout()

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)

        self.input_label = QLabel("Input Directory: Not selected")
        self.output_label = QLabel("Output Directory: Not selected")
        self.status_label = QLabel("Status: Waiting for user action")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.type_censor_label = QLabel("Choose type of censor:")
        self.min_prob_label = QLabel("Set detection sensitivity (default: 0.03 - highest sensitivity, 0.2 - normal,  0.7 - lowest sensitivity)")
        self.metrics_analyze_label = QLabel("Metrics for analysis (set of body parts):")
        self.metrics_analyze_input = QLineEdit(
            ", ".join(censored_classes), parent=self
        )

        self.frame_skip_label = QLabel("How often frames are processed (default: 1). Lower values provide better quality but slower processing")
        self.second_frame_skip_label = QLabel("For example, if set to 1, every frame is processed. If set to 2, every second frame is processed, and so on.")
        self.frame_skip_input = QLineEdit("1",parent=self)

        select_input_button = QPushButton("Select Input Directory")
        select_input_button.clicked.connect(self.select_input_directory)


        self.select_output_button.clicked.connect(self.select_output_directory)

        self.checkbox_auto_output = QCheckBox("Automatically create an output folder (preserving folder structure, created in the program directory)", self)
        self.checkbox_auto_output.stateChanged.connect(self.toggle_auto_output_field)





        self.select_type_censor = QComboBox()
        self.select_type_censor.addItems(["Pixel", "blur", "Black rectangles"])
        self.select_type_censor.currentIndexChanged.connect(self.type_censor_change)



        self.select_min_prob = QComboBox()
        self.select_min_prob.addItems(["0.03","0.05","0.1","0.2","0.3","0.4","0.5","0.7","0.8","0.9"])
        self.select_min_prob.currentIndexChanged.connect(self.change_min_prob)

        process_button = QPushButton("Process Images/video")
        process_button.clicked.connect(self.process_files)
        metrics_info_button = QPushButton("View available metrics", self)
        metrics_info_button.clicked.connect(self.show_available_metrics)





        layout.addWidget(self.input_label)
        layout.addWidget(select_input_button)
        layout.addWidget(self.output_label)
        layout.addWidget(self.select_output_button)

        layout.addWidget(self.checkbox_auto_output)


        layout.addWidget(self.type_censor_label)
        layout.addWidget(self.select_type_censor)

        layout.addWidget(self.min_prob_label)
        layout.addWidget(self.select_min_prob)

        layout.addWidget(self.metrics_analyze_label)
        layout.addWidget(self.metrics_analyze_input)

        layout.addWidget(metrics_info_button)







        layout.addWidget(process_button)
        layout.addWidget(self.status_label)

        layout.addWidget(self.progress_bar)
###################################
        layout.addWidget(self.frame_skip_label)
        layout.addWidget(self.frame_skip_input)
        layout.addWidget(self.second_frame_skip_label)

        self.frame_skip_label.setVisible(False)
        self.frame_skip_input.setVisible(False)  ##### dont SHOW
        self.second_frame_skip_label.setVisible(False)
##################################################
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def show_available_metrics(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Available Metrics")
        dialog.resize(400, 300)  #size

        layout = QVBoxLayout(dialog)
        metrics_list = QListWidget(dialog)
        metrics_list.addItems([
            "FEMALE_GENITALIA_COVERED",
            "FACE_FEMALE",
            "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_BREAST_EXPOSED",
            "ANUS_EXPOSED",
            "FEET_EXPOSED",
            "BELLY_COVERED",
            "FEET_COVERED",
            "ARMPITS_COVERED",
            "ARMPITS_EXPOSED",
            "FACE_MALE",
            "BELLY_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "ANUS_COVERED",
            "FEMALE_BREAST_COVERED",
            "BUTTOCKS_COVERED",
        ])
        metrics_list.itemDoubleClicked.connect(self.handle_item_double_click)

        layout.addWidget(metrics_list)
        dialog.setLayout(layout)
        dialog.show()
        dialog.setAttribute(1, True)

    def handle_item_double_click(self, item):
        try:
            self.add_metric(item.text())
            print(f"add_metric(item.text()) : {item.text()}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


    def add_metric(self, metric):
        """add metrics in censored_classes."""
        global censored_classes
        try:
            if metric in censored_classes:
                QMessageBox.warning(self, "Warning", f"'{metric}' already was added!")
            else:
                censored_classes.append(metric)
                self.metrics_analyze_input.setText(", ".join(censored_classes))
                self.metrics_analyze_input.update()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


    def toggle_auto_output_field(self, state):
        if state == 2:  # Checked
            print(f"Checkbox toggle_auto_output_field was activated..")
            self.output_label.setVisible(False)
            self.select_output_button.setVisible(False)
            self.auto_create_dir_flag = True
        else:
            print(f"Checkbox toggle_auto_output_field was not activated..")
            self.output_label.setVisible(True)
            self.select_output_button.setVisible(True)


    def AutoCreateDir(self):
        unique_name = f"censored_output_dir_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"  # name auto dir
        os.makedirs(os.path.join(BASE_DIR, unique_name), exist_ok=True)
        print(f"Created directory: {os.path.join(BASE_DIR, unique_name)}")
        return os.path.join(BASE_DIR, unique_name)


    def select_input_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_dir = directory
            self.input_label.setText(f"Input Directory: {directory}")
            print(f"Selected directory : {directory}")

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_label.setText(f"Output Directory: {directory}")
            print(f"Selected directory for saved files: {directory}")

    def type_censor_change(self):
        self.type_censor = self.select_type_censor.currentText()
        print(f"Selected type of censor: {self.type_censor}")

    def change_min_prob(self):
        try:
            self.min_prob = float(self.select_min_prob.currentText())
            print(f"The threshold limiter for the probability of the model being triggered has been selected (min_prob): {self.min_prob}")
        except ValueError as e:
            print(f"Error with transformations probability values: {e}")
            self.min_prob = 0.1  # default

    def process_files(self):
        censored_classes = self.metrics_analyze_input.text().split(",")
        text = self.frame_skip_input.text().strip()
        try:
            self.frame_skip = int(text)
            if self.frame_skip < 1:  # Не даем вводить 0 или отрицательные числа
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Frame skip must be a positive integer!")
            self.frame_skip_input.setText("1")  # Возвращаем значение по умолчанию
        print(f"metrics_analyze:", censored_classes)
        print(f"frame_skip:", self.frame_skip)
        if not self.input_dir:
            self.status_label.setText("Status: Please select both directories")
            print("Error: Not be choose directories.")
            return
        self.status_label.setText("Status: Processing images...")
        input_files = [
            f for f in os.listdir(self.input_dir)
            if os.path.isfile(os.path.join(self.input_dir, f)) and f.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.mp4', '.avi', '.mov', '.mkv'))
        ]

        print(f"Found {len(input_files)} files in {self.input_dir}")

        if not input_files:
            self.status_label.setText("Status: No images found in the input directory")
            print("Error: Files not found.")
            return

        print("Begin processing...")
        if self.auto_create_dir_flag is True:
            self.output_dir = self.AutoCreateDir()
        elif not self.output_dir:
            self.output_dir = self.AutoCreateDir()

        total_files = max(1, sum(len(files) for _, _, files in os.walk(self.input_dir)))  # min 1
        self.progress_bar.setMaximum(total_files)
        self.progress_bar.setValue(0)
        processed_files = 0

        for root, dirs, files in os.walk(self.input_dir):
            # Calculate the relative path of the current directory
            rel_path = os.path.relpath(root, self.input_dir)
            # Create the same directory structure in the output directory
            output_subdir = os.path.join(self.output_dir, rel_path)
            os.makedirs(output_subdir, exist_ok=True)

            for file_name in files:
                processed_files += 1
                self.progress_bar.setValue(min(processed_files, total_files))
                input_path = os.path.join(root, file_name)
                output_path = os.path.join(output_subdir, file_name)
                print(f"output_path file:{output_path}")
                print(f"input path file:{input_path}")
                print(f"Processing file: {file_name}")
                try:
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        mode = "photo"
                        image = Image.open(input_path)
                        image_np_photo = np.array(image)
                        detections_photo = self.detector.detect(image_np_photo)
                        censor_image(image, output_path, detections_photo, self.type_censor, mode, self.min_prob)
                    elif file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        mode = "video"
                        censor_video(input_path, output_path, self.detector, self.type_censor, mode, self.min_prob,
                                     self.output_dir,self.frame_skip)
                except Exception as e:
                    print(f"Error with processing file: {file_name}: {e}")
                else:
                    print(f"File {file_name} was processed successfully.")

        self.status_label.setText("Status: Processing completed")
        print("Processing completed.")
#dfg
def censor_video(input_video_path, output_video_path, detector, type_censor, mode, min_prob, output_dir, frame_skip, buffer_size=30):
    cap = cv2.VideoCapture(input_video_path)
    audio_url_file = os.path.join(output_dir, "temp_audio_file.mp3")

    input_video_path = input_video_path.replace("\\", "/")
    audio_url_file = audio_url_file.replace("\\", "/")
    output_video_path = output_video_path.replace("\\", "/")

    print(f"url video: {input_video_path}, url output dir: {output_dir}, audio url: {audio_url_file}")

    extract_audio(input_video_path, audio_url_file)

    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"Video processing: {input_video_path} -> {output_video_path}")

    frame_count = 0
    frame_buffer = []
    censorship_memory = CensorshipMemory(memory_size=4)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_buffer.append(pil_image)

        # Буфер заполнился -> обрабатываем средний кадр и записываем все кадры в порядке поступления
        if len(frame_buffer) == buffer_size:
            middle_index = buffer_size // 2
            target_frame = frame_buffer[middle_index]

            detections = detector.detect(np.array(target_frame))
            if detections:
                censorship_memory.update(detections)

            active_detections = detections if detections else censorship_memory.get_active_detections()

            # Обрабатываем и записываем кадры из буфера, а не только средний
            for i in range(len(frame_buffer)):
                frame_to_write = (
                    censor_image(frame_buffer[i], output_video_path, active_detections, type_censor, mode, min_prob)
                    if active_detections else frame_buffer[i]
                )
                censored_frame_bgr = cv2.cvtColor(np.array(frame_to_write), cv2.COLOR_RGB2BGR)
                out.write(censored_frame_bgr)

            frame_buffer.clear()  # Очистка буфера

    # Записываем оставшиеся кадры
    for frame in frame_buffer:
        detections = detector.detect(np.array(frame))
        if detections:
            censorship_memory.update(detections)

        active_detections = detections if detections else censorship_memory.get_active_detections()
        frame_to_write = (
            censor_image(frame, output_video_path, active_detections, type_censor, mode, min_prob)
            if active_detections else frame
        )
        censored_frame_bgr = cv2.cvtColor(np.array(frame_to_write), cv2.COLOR_RGB2BGR)
        out.write(censored_frame_bgr)

    cap.release()
    out.release()

    combine_video_audio(output_video_path, audio_url_file)
    remover_temp_audio_file(audio_url_file)

    print("Video processing completed.")








def extract_audio(input_video_path, output_audio_path):
    # command ffmpeg
    ffmpeg_command = f'{BASE_DIR}/apps/bin/ffmpeg.exe -i "{input_video_path}" "{output_audio_path}"'
    print(f"ffmpeg_path:{BASE_DIR}/apps/bin/ffmpeg.exe")
    try:
        subprocess.run(ffmpeg_command, shell=True, check=True)
        print(f"Audio extracted to {output_audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")


def remover_temp_audio_file(audio_url_file):
    print(f"removing temp audio... PATH = {audio_url_file}")
    if os.path.exists(audio_url_file):
        os.remove(audio_url_file)
        print(f"Temporary audio file {audio_url_file} removed.")
    else:
        print(f"File {audio_url_file} does not exist, skipping removal.")


def combine_video_audio(processed_video_path, audio_path):

    BASE_DIR = os.path.dirname(__file__)


    if not os.path.exists(processed_video_path):
        raise FileNotFoundError(f"Video file not found: {processed_video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # temp path
    split_video_path = processed_video_path.split("/")
    split_video_path[-1] = "temp_" + split_video_path[-1]
    temp_output_path = "/".join(split_video_path)


    print(f"combine: temp_output_path: {temp_output_path} , processed_video_path : {processed_video_path} ")
    #  FFmpeg
    ffmpeg_command = (
        f'{BASE_DIR}/apps/bin/ffmpeg.exe '
        f'-i "{processed_video_path}" -i "{audio_path}" '
        f'-c:v libx264 -preset slow -crf 18 -c:a aac -b:a 192k -movflags +faststart "{temp_output_path}"'
    )

    try:
        subprocess.run(ffmpeg_command, shell=True, check=True)
        print(f"Temp final file was created: {temp_output_path}")

        # temp file
        if os.path.exists(temp_output_path):
            os.replace(temp_output_path, processed_video_path)
            print(f"final file was saving: {temp_output_path}")
        else:
            print("Error: temp file not will be created.")
    except subprocess.CalledProcessError as e:
        print(f"Error when combining video and audio files: {e}")
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)  # Delete temp file if error
    except Exception as e:
        print(f"Unknown error: {e}")






if __name__ == "__main__":
    print("Running app...")
    app = QApplication([])
    with open("style.qss", "r") as f:
        app.setStyleSheet(f.read())
    window = MainWindow()
    window.show()
    app.exec()
    print("App was completed.")
