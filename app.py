from nemo.collections.asr.models import ASRModel
import torch
import gradio as gr
import spaces
import gc
import shutil
from pathlib import Path
from pydub import AudioSegment
import numpy as np
import os
import gradio.themes as gr_themes
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME="nvidia/parakeet-tdt-0.6b-v2"

model = ASRModel.from_pretrained(model_name=MODEL_NAME)
model.eval()


def start_session(request: gr.Request):
    session_hash = request.session_hash
    session_dir = Path(f'/tmp/{session_hash}')
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"Session with hash {session_hash} started.")
    return session_dir.as_posix()

def end_session(request: gr.Request):
    session_hash = request.session_hash
    session_dir = Path(f'/tmp/{session_hash}')
    
    if session_dir.exists():
        shutil.rmtree(session_dir)

    print(f"Session with hash {session_hash} ended.")

def get_audio_segment(audio_path, start_second, end_second):
    if not audio_path or not Path(audio_path).exists():
        print(f"Warning: Audio path '{audio_path}' not found or invalid for clipping.")
        return None
    try:
        start_ms = int(start_second * 1000)
        end_ms = int(end_second * 1000)

        start_ms = max(0, start_ms)
        if end_ms <= start_ms:
            print(f"Warning: End time ({end_second}s) is not after start time ({start_second}s). Adjusting end time.")
            end_ms = start_ms + 100

        audio = AudioSegment.from_file(audio_path)
        clipped_audio = audio[start_ms:end_ms]

        samples = np.array(clipped_audio.get_array_of_samples())
        if clipped_audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1).astype(samples.dtype)

        frame_rate = clipped_audio.frame_rate
        if frame_rate <= 0:
             print(f"Warning: Invalid frame rate ({frame_rate}) detected for clipped audio.")
             frame_rate = audio.frame_rate

        if samples.size == 0:
             print(f"Warning: Clipped audio resulted in empty samples array ({start_second}s to {end_second}s).")
             return None

        return (frame_rate, samples)
    except FileNotFoundError:
        print(f"Error: Audio file not found at path: {audio_path}")
        return None
    except Exception as e:
        print(f"Error clipping audio {audio_path} from {start_second}s to {end_second}s: {e}")
        return None

@spaces.GPU
def get_transcripts_and_raw_times(audio_path, session_dir):
    if not audio_path:
        gr.Error("No audio file path provided for transcription.", duration=None)
        # Return an update to hide the button
        return [], [], None, gr.DownloadButton(visible=False)

    vis_data = [["N/A", "N/A", "Processing failed"]]
    raw_times_data = [[0.0, 0.0]]
    processed_audio_path = None
    csv_file_path = None
    original_path_name = Path(audio_path).name
    audio_name = Path(audio_path).stem

    try:
        try:
            gr.Info(f"Loading audio: {original_path_name}", duration=2)
            audio = AudioSegment.from_file(audio_path)
        except Exception as load_e:
            gr.Error(f"Failed to load audio file {original_path_name}: {load_e}", duration=None)
            # Return an update to hide the button
            return [["Error", "Error", "Load failed"]], [[0.0, 0.0]], audio_path, gr.DownloadButton(visible=False)

        resampled = False
        mono = False

        target_sr = 16000
        if audio.frame_rate != target_sr:
            try:
                audio = audio.set_frame_rate(target_sr)
                resampled = True
            except Exception as resample_e:
                 gr.Error(f"Failed to resample audio: {resample_e}", duration=None)
                 # Return an update to hide the button
                 return [["Error", "Error", "Resample failed"]], [[0.0, 0.0]], audio_path, gr.DownloadButton(visible=False)

        if audio.channels == 2:
            try:
                audio = audio.set_channels(1)
                mono = True
            except Exception as mono_e:
                 gr.Error(f"Failed to convert audio to mono: {mono_e}", duration=None)
                 # Return an update to hide the button
                 return [["Error", "Error", "Mono conversion failed"]], [[0.0, 0.0]], audio_path, gr.DownloadButton(visible=False)
        elif audio.channels > 2:
             gr.Error(f"Audio has {audio.channels} channels. Only mono (1) or stereo (2) supported.", duration=None)
             # Return an update to hide the button
             return [["Error", "Error", f"{audio.channels}-channel audio not supported"]], [[0.0, 0.0]], audio_path, gr.DownloadButton(visible=False)

        if resampled or mono:
            try:
                processed_audio_path = Path(session_dir, f"{audio_name}_resampled.wav")
                audio.export(processed_audio_path, format="wav")
                transcribe_path = processed_audio_path.as_posix()
                info_path_name = f"{original_path_name} (processed)"
            except Exception as export_e:
                gr.Error(f"Failed to export processed audio: {export_e}", duration=None)
                if processed_audio_path and os.path.exists(processed_audio_path):
                    os.remove(processed_audio_path)
                # Return an update to hide the button
                return [["Error", "Error", "Export failed"]], [[0.0, 0.0]], audio_path, gr.DownloadButton(visible=False)
        else:
            transcribe_path = audio_path
            info_path_name = original_path_name

        try:
            model.to(device)
            gr.Info(f"Transcribing {info_path_name} on {device}...", duration=2)
            output = model.transcribe([transcribe_path], timestamps=True)

            if not output or not isinstance(output, list) or not output[0] or not hasattr(output[0], 'timestamp') or not output[0].timestamp or 'segment' not in output[0].timestamp:
                 gr.Error("Transcription failed or produced unexpected output format.", duration=None)
                 # Return an update to hide the button
                 return [["Error", "Error", "Transcription Format Issue"]], [[0.0, 0.0]], audio_path, gr.DownloadButton(visible=False)

            segment_timestamps = output[0].timestamp['segment']
            csv_headers = ["Start (s)", "End (s)", "Segment"]
            vis_data = [[f"{ts['start']:.2f}", f"{ts['end']:.2f}", ts['segment']] for ts in segment_timestamps]
            raw_times_data = [[ts['start'], ts['end']] for ts in segment_timestamps]

            # Default button update (hidden) in case CSV writing fails
            button_update = gr.DownloadButton(visible=False)
            try:
                csv_file_path = Path(session_dir, f"transcription_{audio_name}.csv")
                writer = csv.writer(open(csv_file_path, 'w'))
                writer.writerow(csv_headers)
                writer.writerows(vis_data)
                print(f"CSV transcript saved to temporary file: {csv_file_path}")
                # If CSV is saved, create update to show button with path
                button_update = gr.DownloadButton(value=csv_file_path, visible=True)
            except Exception as csv_e:
                gr.Error(f"Failed to create transcript CSV file: {csv_e}", duration=None)
                print(f"Error writing CSV: {csv_e}")
                # csv_file_path remains None, button_update remains hidden

            gr.Info("Transcription complete.", duration=2)
            # Return the data and the button update dictionary
            return vis_data, raw_times_data, audio_path, button_update

        except torch.cuda.OutOfMemoryError as e:
            error_msg = 'CUDA out of memory. Please try a shorter audio or reduce GPU load.'
            print(f"CUDA OutOfMemoryError: {e}")
            gr.Error(error_msg, duration=None)
            # Return an update to hide the button
            return [["OOM", "OOM", error_msg]], [[0.0, 0.0]], audio_path, gr.DownloadButton(visible=False)

        except FileNotFoundError:
            error_msg = f"Audio file for transcription not found: {Path(transcribe_path).name}."
            print(f"Error: Transcribe audio file not found at path: {transcribe_path}")
            gr.Error(error_msg, duration=None)
            # Return an update to hide the button
            return [["Error", "Error", "File not found for transcription"]], [[0.0, 0.0]], audio_path, gr.DownloadButton(visible=False)

        except Exception as e:
            error_msg = f"Transcription failed: {e}"
            print(f"Error during transcription processing: {e}")
            gr.Error(error_msg, duration=None)
            vis_data = [["Error", "Error", error_msg]]
            raw_times_data = [[0.0, 0.0]]
            # Return an update to hide the button
            return vis_data, raw_times_data, audio_path, gr.DownloadButton(visible=False)
        finally:
            try:
                if 'model' in locals() and hasattr(model, 'cpu'):
                     if device == 'cuda':
                          model.cpu()
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as cleanup_e:
                print(f"Error during model cleanup: {cleanup_e}")
                gr.Warning(f"Issue during model cleanup: {cleanup_e}", duration=5)

    finally:
        if processed_audio_path and os.path.exists(processed_audio_path):
            try:
                os.remove(processed_audio_path)
                print(f"Temporary audio file {processed_audio_path} removed.")
            except Exception as e:
                print(f"Error removing temporary audio file {processed_audio_path}: {e}")

def play_segment(evt: gr.SelectData, raw_ts_list, current_audio_path):
    if not isinstance(raw_ts_list, list):
        print(f"Warning: raw_ts_list is not a list ({type(raw_ts_list)}). Cannot play segment.")
        return gr.Audio(value=None, label="Selected Segment")

    if not current_audio_path:
        print("No audio path available to play segment from.")
        return gr.Audio(value=None, label="Selected Segment")

    selected_index = evt.index[0]

    if selected_index < 0 or selected_index >= len(raw_ts_list):
         print(f"Invalid index {selected_index} selected for list of length {len(raw_ts_list)}.")
         return gr.Audio(value=None, label="Selected Segment")

    if not isinstance(raw_ts_list[selected_index], (list, tuple)) or len(raw_ts_list[selected_index]) != 2:
         print(f"Warning: Data at index {selected_index} is not in the expected format [start, end].")
         return gr.Audio(value=None, label="Selected Segment")

    start_time_s, end_time_s = raw_ts_list[selected_index]

    print(f"Attempting to play segment: {current_audio_path} from {start_time_s:.2f}s to {end_time_s:.2f}s")

    segment_data = get_audio_segment(current_audio_path, start_time_s, end_time_s)

    if segment_data:
        print("Segment data retrieved successfully.")
        return gr.Audio(value=segment_data, autoplay=True, label=f"Segment: {start_time_s:.2f}s - {end_time_s:.2f}s", interactive=False)
    else:
        print("Failed to get audio segment data.")
        return gr.Audio(value=None, label="Selected Segment")

article = (
    "<p style='font-size: 1.1em;'>"
    "This demo showcases <code><a href='https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2'>parakeet-tdt-0.6b-v2</a></code>, a 600-million-parameter model designed for high-quality English speech recognition."
    "</p>"
    "<p><strong style='color: red; font-size: 1.2em;'>Key Features:</strong></p>"
    "<ul style='font-size: 1.1em;'>"
    "    <li>Automatic punctuation and capitalization</li>"
    "    <li>Accurate word-level timestamps (click on a segment in the table below to play it!)</li>"
    "    <li>Efficiently transcribes long audio segments (up to 20 minutes) <small>(For even longer audios, see <a href='https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_infer_rnnt.py' target='_blank'>this script</a>)</small></li>"
    "    <li>Robust performance on spoken numbers, and song lyrics transcription </li>"
    "</ul>"
    "<p style='font-size: 1.1em;'>"
    "This model is <strong>available for commercial and non-commercial use</strong>."
    "</p>"
    "<p style='text-align: center;'>"
    "<a href='https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2' target='_blank'>🎙️ Learn more about the Model</a> | "
    "<a href='https://arxiv.org/abs/2305.05084' target='_blank'>📄 Fast Conformer paper</a> | "
    "<a href='https://arxiv.org/abs/2304.06795' target='_blank'>📚 TDT paper</a> | "
    "<a href='https://github.com/NVIDIA/NeMo' target='_blank'>🧑‍💻 NeMo Repository</a>"
    "</p>"
)

examples = [
    ["data/example-yt_saTD1u8PorI.mp3"],
]

# Define an NVIDIA-inspired theme
nvidia_theme = gr_themes.Default(
    primary_hue=gr_themes.Color(
        c50="#E6F1D9", # Lightest green
        c100="#CEE3B3",
        c200="#B5D58C",
        c300="#9CC766",
        c400="#84B940",
        c500="#76B900", # NVIDIA Green
        c600="#68A600",
        c700="#5A9200",
        c800="#4C7E00",
        c900="#3E6A00", # Darkest green
        c950="#2F5600"
    ),
    neutral_hue="gray", # Use gray for neutral elements
    font=[gr_themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set()

# Apply the custom theme
with gr.Blocks(theme=nvidia_theme) as demo:
    model_display_name = MODEL_NAME.split('/')[-1] if '/' in MODEL_NAME else MODEL_NAME
    gr.Markdown(f"<h1 style='text-align: center; margin: 0 auto;'>Speech Transcription with {model_display_name}</h1>")
    gr.HTML(article)

    current_audio_path_state = gr.State(None)
    raw_timestamps_list_state = gr.State([])

    session_dir = gr.State()
    demo.load(start_session, outputs=[session_dir])

    with gr.Tabs():
        with gr.TabItem("Audio File"):
            file_input = gr.Audio(sources=["upload"], type="filepath", label="Upload Audio File")
            gr.Examples(examples=examples, inputs=[file_input], label="Example Audio Files (Click to Load)")
            file_transcribe_btn = gr.Button("Transcribe Uploaded File", variant="primary")
        
        with gr.TabItem("Microphone"):
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Audio")
            mic_transcribe_btn = gr.Button("Transcribe Microphone Input", variant="primary")

    gr.Markdown("---")
    gr.Markdown("<p><strong style='color: #FF0000; font-size: 1.2em;'>Transcription Results (Click row to play segment)</strong></p>")

    # Define the DownloadButton *before* the DataFrame
    download_btn = gr.DownloadButton(label="Download Transcript (CSV)", visible=False)

    vis_timestamps_df = gr.DataFrame(
        headers=["Start (s)", "End (s)", "Segment"],
        datatype=["number", "number", "str"],
        wrap=True,
        label="Transcription Segments"
    )

    # selected_segment_player was defined after download_btn previously, keep it after df for layout
    selected_segment_player = gr.Audio(label="Selected Segment", interactive=False)

    mic_transcribe_btn.click(
        fn=get_transcripts_and_raw_times,
        inputs=[mic_input, session_dir],
        outputs=[vis_timestamps_df, raw_timestamps_list_state, current_audio_path_state, download_btn],
        api_name="transcribe_mic"
    )

    file_transcribe_btn.click(
        fn=get_transcripts_and_raw_times,
        inputs=[file_input, session_dir],
        outputs=[vis_timestamps_df, raw_timestamps_list_state, current_audio_path_state, download_btn],
        api_name="transcribe_file"
    )

    vis_timestamps_df.select(
        fn=play_segment,
        inputs=[raw_timestamps_list_state, current_audio_path_state],
        outputs=[selected_segment_player],
    )

    demo.unload(end_session)

if __name__ == "__main__":
    print("Launching Gradio Demo...")
    demo.queue()
    demo.launch()