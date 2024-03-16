import speech_recognition as sr
from moviepy.editor import VideoFileClip
from pocketsphinx import LiveSpeech, get_model_path, AudioFile
import os
# 定义一个函数，用于将音频文件转换为文字
def convert_audio_to_text(audio_file):
    # 创建一个识别器实例
    recognizer = sr.Recognizer()
    # 使用麦克风作为音频输入源（需要麦克风权限）
    with sr.AudioFile(audio_file) as source:
        # 读取音频文件
        # audio_listened = recognizer.listen(source)
        studio = recognizer.record(source)
        try:
            # 使用Google语音识别
            text = recognizer.recognize_sphinx(studio, language='zh-CN', show_all=True)
            print(text.seg())
            return str(text)
        except sr.UnknownValueError:
            print("无法理解音频")
        except sr.RequestError:
            print("无法请求结果")

config = {
    'verbose': False,
    'audio_file': 'audio.wav',
    'hmm': get_model_path('zh-CN'),
    'lm': get_model_path('en-us.lm.bin'),
    'dict': get_model_path('cmudict-en-us.dict')
}

def sphinx_convert(audio_file):
    fps = 30
    for phrase in AudioFile(audio_file):  # frate (default=100)
        print('-' * 28)
        print('| %5s |  %3s  |   %4s   |' % ('start', 'end', 'word'))
        print('-' * 28)
        for s in phrase.seg():
            print('| %4ss | %4ss | %8s |' % (s.start_frame / fps, s.end_frame / fps, s.word))
        print('-' * 28)

# 定义一个函数，用于从视频文件中提取音频
def extract_audio_from_video(video_file, audio_file):
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(audio_file, codec="pcm_s16le")

def whisper_convert(audio_file, text_folder):
    import subprocess
    p = subprocess.Popen("whisper " + audio_file + " -o " + text_folder, shell=True)
    res = p.returncode
    p.wait()


def main():
    video_folder = "/public/home/huangshisheng/gaussian-splatting/endingDesign/data_raw/"
    audio_folder = "/public/home/huangshisheng/gaussian-splatting/endingDesign/audio2txt/"
    finish = set()
    for root,dirs,files in os.walk(audio_folder):
        for name in files:
            if name.endswith(".txt"):
                finish.add(name[:-4])
    for root,dirs,files in os.walk(video_folder):
        # print(root, dirs, files)
        for name in files:
            if name.endswith(".wmv") and name[:-4] not in finish:
                video_file = os.path.join(root,name)
                audio_root = root.replace("data_raw", "audio2txt")
                if  os.path.exists(audio_root) is False:
                    os.mkdir(audio_root)
                audio_file = os.path.join(audio_root, name[:-4] + ".wav")
                text_file = os.path.join(audio_root, name[:-4]+".txt")
                # text_folder = root
                # 从视频文件中提取音频
                print(video_file, "to", audio_file)
                extract_audio_from_video(video_file, audio_file)
                # 将音频文件转换为文字
                print(audio_file, "to txt")
                whisper_convert(audio_file, root)

if __name__ == "__main__":
    main()