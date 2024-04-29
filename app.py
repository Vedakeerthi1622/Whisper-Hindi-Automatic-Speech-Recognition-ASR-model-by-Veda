import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2ProcessorWithLM
import gradio as gr
import sox
import subprocess


def read_file_and_process(wav_file):
    filename = wav_file.split('.')[0]
    filename_16k = filename + "16k.wav"
    resampler(wav_file, filename_16k)
    speech, _ = sf.read(filename_16k)
    inputs = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)
    
    return inputs


def resampler(input_file_path, output_file_path):
    command = (
        f"ffmpeg -hide_banner -loglevel panic -i {input_file_path} -ar 16000 -ac 1 -bits_per_raw_sample 16 -vn "
        f"{output_file_path}"
    )
    subprocess.call(command, shell=True)


def parse_transcription_with_lm(logits):
    result = processor_with_LM.batch_decode(logits.cpu().numpy())
    text = result.text
    transcription = text[0].replace('<s>','')
    return transcription

def parse_transcription(logits):
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    return transcription

def parse(wav_file, applyLM):
    input_values = read_file_and_process(wav_file)
    with torch.no_grad():
        logits = model(**input_values).logits
   
    if applyLM:
        return parse_transcription_with_lm(logits)
    else:
        return parse_transcription(logits)

    
model_id = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
processor = Wav2Vec2Processor.from_pretrained(model_id)
processor_with_LM = Wav2Vec2ProcessorWithLM.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

    
input_ = gr.Audio(source="microphone", type="filepath") 
txtbox = gr.Textbox(
            label="Output from model will appear here:",
            lines=5
        )
chkbox = gr.Checkbox(label="Apply LM", value=False)


gr.Interface(parse, inputs = [input_, chkbox],  outputs=txtbox,
             streaming=True, interactive=True,
             analytics_enabled=False, show_tips=False, enable_queue=True).launch(inline=False);
