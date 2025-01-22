import os
from nudenet import NudeDetector
from PIL import Image, ImageDraw, ImageFilter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog, QLabel, QWidget, QComboBox, QLineEdit
)
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import subprocess
import ffmpeg

# classes list
censored_classes = ['FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED', 'BUTTOCKS_EXPOSED']
BASE_DIR = os.path.dirname(__file__)

def censor_image(image, output_path, detector,type_censor,mode,min_prob):
    print(f"Обрабатываем изображение: {image}")
    print(f"Тип цензуры: {type_censor}")
    image_np = np.array(image)

    detections = detector.detect(image_np)
    # detections = detector.detect(image)
    print(f"Ответ модели для изображения {image}: {detections}")
    if not detections:
        print(f"Для изображения {image} не было найдено интимных мест.")

    filtered_detections = [
        detection for detection in detections
        if detection['score'] >= min_prob
    ]
    print(f"Фильтрованные детекции: {filtered_detections}")


    draw = ImageDraw.Draw(image)


    if type_censor == "Black rectangles":
        print(f"Тип цензуры: Black rectangles")
        try:

            if isinstance(filtered_detections, list):
                for detection in filtered_detections:
                    print(f"Детекция: {detection}")
                    if detection['class'] in censored_classes:
                        x1, y1, x2, y2 = detection['box']
                        draw.rectangle([x1, y1, x1 + x2, y1 + y2], fill="black")
            else:
                print(f"Неверная структура данных для изображения {image}")
            if (mode == "video"):
                return image
            else:
                image.save(output_path)
                print(f"Изображение сохранено как: {output_path}")
        except Exception as e:
            print(f"Ошибка при обработке {image}: {e}")
    elif type_censor == "Pixel":
        print(f"Тип цензуры: Pixel")
        try:

            if isinstance(filtered_detections, list):
                for detection in filtered_detections:
                    print(f"Детекция: {detection}")
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
                print(f"Неверная структура данных для изображения {image}")
            if(mode == "video"):
                return image
            else:
                image.save(output_path)
                print(f"Изображение сохранено как: {output_path}")
        except Exception as e:
            print(f"Ошибка при обработке {image}: {e}")
    elif type_censor == "blur":
        print(f"Тип цензуры: blur")
        #self.label_blur_size = QLabel("Blur power: (ONLY odd number, default - 55)")
        #self.select_blur_size = QLineEdit("50", parent=self)
        #size_blur = int(self.select_blur_size.text())
        #self.layout.addWidget(self.label_blur_size)
        #self.layout.addWidget(self.select_blur_size)

        try:
            if isinstance(filtered_detections, list):
                for detection in filtered_detections:
                    print(f"Детекция: {detection}")
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
                print(f"Неверная структура данных для изображения {image}")
            if(mode == "video"):
                return image
            else:
                image.save(output_path)
                print(f"Изображение сохранено как: {output_path}")
        except Exception as e:
            print(f"Ошибка при обработке {image}: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.detector = NudeDetector()
        self.input_dir = None
        self.output_dir = None
        self.type_censor = "Pixel"
        self.min_prob = 0.1

    def init_ui(self):
        self.setWindowTitle("Censorship Tool")
        self.setGeometry(300, 300, 400, 200)

        layout = QVBoxLayout()

        self.input_label = QLabel("Input Directory: Not selected")
        self.output_label = QLabel("Output Directory: Not selected")
        self.status_label = QLabel("Status: Waiting for user action")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.type_censor_label = QLabel("Choose type of censor:")
        self.min_prob_label = QLabel("Choose probability score limit (default = 0.1 - maximum model reaction)")
        self.metrics_analyze_label = QLabel("Analyze metrics:")
        self.metrics_analyze_input = QLineEdit(
            "FEMALE_BREAST_EXPOSED, FEMALE_GENITALIA_EXPOSED, BUTTOCKS_EXPOSED", parent=self
        )


        select_input_button = QPushButton("Select Input Directory")
        select_input_button.clicked.connect(self.select_input_directory)

        select_output_button = QPushButton("Select Output Directory")
        select_output_button.clicked.connect(self.select_output_directory)

        self.select_type_censor = QComboBox()
        self.select_type_censor.addItems(["Pixel", "blur", "Black rectangles"])
        self.select_type_censor.currentIndexChanged.connect(self.type_censor_change)





        self.select_min_prob = QComboBox()
        self.select_min_prob.addItems(["0.1","0.2","0.3","0.4","0.5","0.7","0.8","0.9"])
        self.select_min_prob.currentIndexChanged.connect(self.change_min_prob)

        process_button = QPushButton("Process Images/video")
        process_button.clicked.connect(self.process_files)

        layout.addWidget(self.input_label)
        layout.addWidget(select_input_button)
        layout.addWidget(self.output_label)
        layout.addWidget(select_output_button)

        layout.addWidget(self.type_censor_label)
        layout.addWidget(self.select_type_censor)

        layout.addWidget(self.min_prob_label)
        layout.addWidget(self.select_min_prob)

        layout.addWidget(process_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.metrics_analyze_label)
        layout.addWidget(self.metrics_analyze_input)


        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select_input_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_dir = directory
            self.input_label.setText(f"Input Directory: {directory}")
            print(f"Выбрана директория : {directory}")

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_label.setText(f"Output Directory: {directory}")
            print(f"Выбрана директория для сохранения: {directory}")

    def type_censor_change(self):
        self.type_censor = self.select_type_censor.currentText()
        print(f"Выбран тип цензуры: {self.type_censor}")

    def change_min_prob(self):
        try:
            self.min_prob = float(self.select_min_prob.currentText())
            print(f"Выбран пороговый ограничитель вероятности срабатывания модели: {self.min_prob}")
        except ValueError as e:
            print(f"Ошибка преобразования значения вероятности: {e}")
            self.min_prob = 0.1  # default


    def process_files(self):
        censored_classes = self.metrics_analyze_input.text().split(",")
        print(f"Выбранные метрики:",censored_classes)
        if not self.input_dir or not self.output_dir:
            self.status_label.setText("Status: Please select both directories")
            print("Ошибка: Не выбраны обе директории.")
            return
        self.status_label.setText("Status: Processing images...")
        input_files = [
            f for f in os.listdir(self.input_dir)
            if os.path.isfile(os.path.join(self.input_dir, f)) and f.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.mp4', '.avi', '.mov', '.mkv'))
        ]

        print(f"Найдено {len(input_files)} файлов в директории {self.input_dir}")

        if not input_files:
            self.status_label.setText("Status: No images found in the input directory")
            print("Ошибка: В директории не найдено файлов.")
            return

        print("Начинаем обработку...")

        for file_name in input_files:
            input_path = os.path.join(self.input_dir, file_name)
            output_path = os.path.join(self.output_dir, file_name)
            print(f"Обрабатываем файл {file_name}")
            try:
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    mode = "photo"
                    image = Image.open(input_path)
                    censor_image(image, output_path, self.detector, self.type_censor, mode, self.min_prob)
                elif file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    mode = "video"
                    censor_video(input_path, output_path, self.detector, self.type_censor, mode, self.min_prob,self.output_dir)
            except Exception as e:
                print(f"Ошибка при обработке файла {file_name}: {e}")
            else:
                print(f"Файл {file_name} успешно обработан.")

        self.status_label.setText("Status: Processing completed")
        print("Обработка завершена.")


def censor_video(input_video_path, output_video_path, detector, type_censor,mode,min_prob,output_dir,frame_skip=1,buffer_size=200): # frame_skip=1 - каждый кадр обрабатывается, frame_skip=2 - каждый второй и т.д.
    cap = cv2.VideoCapture(input_video_path)
    audio_url_file = os.path.join(output_dir, "temp_audio_file.mp3")
    input_video_path = input_video_path.replace("\\", "/")
    audio_url_file = audio_url_file.replace("\\", "/")
    output_video_path = output_video_path.replace("\\", "/")

    print(f"url video: {input_video_path}, url output dir:{output_dir}, audio url : {audio_url_file}")

    extract_audio(input_video_path, audio_url_file)


    if not cap.isOpened():
        print(f"Ошибка при открытии видео: {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Кодек для сохранения видео в формате .avi
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"Обработка видео: {input_video_path} -> {output_video_path}")

    frame_count = 0
    frame_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % frame_skip != 0:
            out.write(frame)
            continue


        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_buffer.append(pil_image)

        if len(frame_buffer) == buffer_size:

            middle_index = buffer_size // 2
            target_frame = frame_buffer[middle_index]
            censored_frame = censor_image(target_frame, output_video_path, detector, type_censor, mode,min_prob)

            # Преобразуем кадр обратно в  OpenCV
            censored_frame_bgr = cv2.cvtColor(np.array(censored_frame), cv2.COLOR_RGB2BGR)

            # Записываем кадр в видео
            out.write(censored_frame_bgr)
            frame_buffer.pop(0)
    while len(frame_buffer) > 0:
        target_frame = frame_buffer.pop(0)
        censored_frame = censor_image(target_frame, output_video_path, detector, type_censor, mode,min_prob)
        censored_frame_bgr = cv2.cvtColor(np.array(censored_frame), cv2.COLOR_RGB2BGR)
        out.write(censored_frame_bgr)

    cap.release()
    out.release()
    combine_video_audio(output_video_path,audio_url_file,output_video_path)
    remover_temp_audio_file(audio_url_file)



    print("Обработка видео завершена.")


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


def combine_video_audio(processed_video_path, audio_path, output_video_path):

    BASE_DIR = os.path.dirname(__file__)


    if not os.path.exists(processed_video_path):
        raise FileNotFoundError(f"Видео файл не найден: {processed_video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")

    # temp path
    temp_output_path = output_video_path.replace(".mp4", "_temp.mp4")

    #  FFmpeg
    ffmpeg_command = (
        f'{BASE_DIR}/apps/bin/ffmpeg.exe '
        f'-i "{processed_video_path}" -i "{audio_path}" '
        f'-c:v libx264 -preset slow -crf 18 -c:a aac -b:a 192k -movflags +faststart "{temp_output_path}"'
    )

    try:
        subprocess.run(ffmpeg_command, shell=True, check=True)
        print(f"Временный файл с объединённым видео и аудио сохранён: {temp_output_path}")

        # temp file
        if os.path.exists(temp_output_path):
            os.replace(temp_output_path, output_video_path)
            print(f"Итоговый файл сохранён: {output_video_path}")
        else:
            print("Ошибка: временный файл не был создан.")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при объединении видео и аудио: {e}")
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)  # Удаляем временный файл в случае ошибки
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")






if __name__ == "__main__":
    print("Запуск приложения...")
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
    print("Приложение завершило работу.")
