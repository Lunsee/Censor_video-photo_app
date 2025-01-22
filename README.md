by Lunsee  <3

# Steps to Set Up the Project

## 1. Clone the Repository

To get a local copy of the project, run:
```bash
git clone <repository_url>
cd <repository_name>
```

## 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```
## 3 . Install Dependencies

Install the required dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```
## 4. Run the Project

Start the project using the main script:
```bash
python <main_script>.py

Replace <main_script>.py with the actual name of the script you want to run.
```

## Check carefully that ffmpeg.exe it was also installed so that the path in the program was correctly specified



### Install FFmpeg

1. **Download FFmpeg**  
   To get started, download the FFmpeg executable file for your operating system from [FFmpeg official website](https://ffmpeg.org/download.html ). Select the appropriate version (for example, download the `.exe` file for Windows).

2. **Install FFmpeg**  
   After downloading, unzip the archive and install FFmpeg in a convenient location on your computer.

3. **Specify the path in the program**  
   In the program, you need to specify the path to the executable file `ffmpeg.exe `. Use the following code to configure the path:

```python
# FFmpeg
ffmpeg_command = (
    f'{BASE_DIR}/apps/bin/ffmpeg.exe '  # Убедитесь, что путь к ffmpeg.exe правильный
    f'-i "{processed_video_path}" -i "{audio_path}" '
    f'-c:v libx264 -preset slow -crf 18 -c:a aac -b:a 192k -movflags +faststart "{temp_output_path}"'
)










1. **Скачайте FFmpeg**  
   Для начала скачайте исполняемый файл FFmpeg для вашей операционной системы с [официального сайта FFmpeg](https://ffmpeg.org/download.html). Выберите подходящую версию (например, для Windows скачайте файл `.exe`).

2. **Установите FFmpeg**  
   После скачивания распакуйте архив и установите FFmpeg в удобное место на вашем компьютере.

3. **Укажите путь в программе**  
   В программе вам нужно указать путь к исполняемому файлу `ffmpeg.exe`. Используйте следующий код для конфигурации пути:

```python
# FFmpeg
ffmpeg_command = (
    f'{BASE_DIR}/apps/bin/ffmpeg.exe '  # Убедитесь, что путь к ffmpeg.exe правильный
    f'-i "{processed_video_path}" -i "{audio_path}" '
    f'-c:v libx264 -preset slow -crf 18 -c:a aac -b:a 192k -movflags +faststart "{temp_output_path}"'
)




















