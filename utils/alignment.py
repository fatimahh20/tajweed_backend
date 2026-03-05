import os
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
from phonemizer.backend.espeak.wrapper import EspeakWrapper


#----------------------------------------------------------
# 1. Point to the libespeak-ng.dll file
library_path = r'C:\Program Files\eSpeak NG\libespeak-ng.dll'
EspeakWrapper.set_library(library_path)

# 2. Update the system PATH so Python 3.11 can find the dependencies
espeak_path = r'C:\Program Files\eSpeak NG'
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(espeak_path)
#----------------------------------------------------------

# Define the absolute path to your eSpeak installatio
import os

# Only set Windows paths if on Windows
if os.name == 'nt':
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    library_path = r'C:\Program Files\eSpeak NG\libespeak-ng.dll'
    EspeakWrapper.set_library(library_path)
    os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG"
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)
model.eval()

# Decoder Setup from your code
labels = processor.tokenizer.convert_ids_to_tokens(list(range(processor.tokenizer.vocab_size)))
decoder = build_ctcdecoder(labels=labels, kenlm_model_path=None)

def forced_align(audio_path):
    """EXACT function from your Colab"""
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits

    logits_numpy = logits.cpu().detach().numpy()[0]
    beam_decoded_string = decoder.decode(logits_numpy, beam_width=10)

    predicted_ids = torch.argmax(logits, dim=-1).squeeze()
    num_frames = logits.shape[1]
    detected_phonemes = beam_decoded_string.split()

    return predicted_ids.cpu().numpy(), num_frames, detected_phonemes