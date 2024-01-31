from nemo.collections.asr.models import ASRModel
import yt_dlp as youtube_dl
import os
import tempfile
import torch
import gradio as gr
from pydub import AudioSegment

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME="nvidia/parakeet-tdt-1.1b"
YT_LENGTH_LIMIT_S=3600

model = ASRModel.from_pretrained(model_name=MODEL_NAME).to(device)
model.eval()

def get_transcripts(audio_path):
    text = model.transcribe([audio_path])[0][0]
    return text

article = (
    "<p style='text-align: center'>"
    "<a href='https://huggingface.co/nvidia/parakeet-tdt-1.1b' target='_blank'>🎙️ Learn more about Parakeet TDT model</a> | "
    "<a href='https://arxiv.org/abs/2304.06795' target='_blank'>📚 TDT ICML paper</a> | "
    "<a href='https://github.com/NVIDIA/NeMo' target='_blank'>🧑‍💻 Repository</a>"
    "</p>"
)
examples = [
    ["data/conversation.wav"],
    ["data/id10270_5r0dWxy17C8-00001.wav"],
]

def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str

def download_yt_audio(yt_url, filename):
    info_loader = youtube_dl.YoutubeDL()
    
    try:
        info = info_loader.extract_info(yt_url, download=False)
    except youtube_dl.utils.DownloadError as err:
        raise gr.Error(str(err))
    
    file_length = info["duration_string"]
    file_h_m_s = file_length.split(":")
    file_h_m_s = [int(sub_length) for sub_length in file_h_m_s]
    
    if len(file_h_m_s) == 1:
        file_h_m_s.insert(0, 0)
    if len(file_h_m_s) == 2:
        file_h_m_s.insert(0, 0)
    file_length_s = file_h_m_s[0] * 3600 + file_h_m_s[1] * 60 + file_h_m_s[2]
    
    if file_length_s > YT_LENGTH_LIMIT_S:
        yt_length_limit_hms = time.strftime("%HH:%MM:%SS", time.gmtime(YT_LENGTH_LIMIT_S))
        file_length_hms = time.strftime("%HH:%MM:%SS", time.gmtime(file_length_s))
        raise gr.Error(f"Maximum YouTube length is {yt_length_limit_hms}, got {file_length_hms} YouTube video.")
    
    ydl_opts = {"outtmpl": filename, "format": "worstvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"}
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([yt_url])
        except youtube_dl.utils.ExtractorError as err:
            raise gr.Error(str(err))


def yt_transcribe(yt_url, max_filesize=75.0):
    html_embed_str = _return_yt_html_embed(yt_url)

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "video.mp4")
        download_yt_audio(yt_url, filepath)
        audio = AudioSegment.from_file(filepath)
        wav_filepath = os.path.join(tmpdirname, "audio.wav")
        audio.export(wav_filepath, format="wav")

    text = get_transcripts(wav_filepath)
    return html_embed_str, text


demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=get_transcripts,
    inputs=[
        gr.Audio(sources="microphone", type="filepath")
    ],
    outputs="text",
    theme="huggingface",
    title="Parakeet TDT 1.1B: Transcribe Audio",
    description=(
        "Transcribe microphone or audio inputs with the click of a button! Demo uses the"
        f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) to transcribe audio files"
        " of arbitrary length. TDT models are 75% more efficient than similar size RNNT model"
    ),
    allow_flagging="never",
)

file_transcribe = gr.Interface(
    fn=get_transcripts,
    inputs=[
        gr.Audio(sources="upload", type="filepath", label="Audio file"),
    ],
    outputs="text",
    theme="huggingface",
    title="Parakeet TDT 1.1B: Transcribe Audio",
    description=(
        "Transcribe microphone or audio inputs with the click of a button! Demo uses the"
        f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) to transcribe audio files"
        " of arbitrary length. TDT models are 75% more efficient than similar size RNNT model"
    ),
    allow_flagging="never",
)

youtube_transcribe = gr.Interface(
    fn=yt_transcribe,
    inputs=[
        gr.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL"),
    ],
    outputs=["html", "text"],
    theme="huggingface",
    title="Parakeet TDT 1.1B: Transcribe Audio",
    description=(
        "Transcribe microphone or audio inputs with the click of a button! Demo uses the"
        f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) to transcribe audio files"
        " of arbitrary length. TDT models are 75% more efficient than similar size RNNT model"
    ),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([mf_transcribe, file_transcribe], ["Microphone", "Audio file"])

demo.launch()
