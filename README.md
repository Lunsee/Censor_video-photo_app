by Lunsee  <3

# Description 
This program is a censorship tool that processes images and videos to censor explicit content.
It uses the nudenet model for detecting exposed intimate body parts and applies various censorship methods, including black rectangles, pixelation, and blurring.

The program uses the NudeDetector artificial intelligence model from the nudenet library to detect intimate areas in images and videos. The model analyzes the images, determining whether there are naked areas of the body on them, such as women's breasts, genitals, and buttocks (You can provide metrics for censorship yourself).

Metrics:
```python
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
```

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
    f'{BASE_DIR}/apps/bin/ffmpeg.exe '  #  ffmpeg.exe PATH !!!!
    f'-i "{processed_video_path}" -i "{audio_path}" '
    f'-c:v libx264 -preset slow -crf 18 -c:a aac -b:a 192k -movflags +faststart "{temp_output_path}"'
)
```




















