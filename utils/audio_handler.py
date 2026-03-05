import torch
import librosa
import numpy as np
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2FeatureExtractor

# --- STEP 1: DEFINE GLOBALS AT MODULE LEVEL ---
_processor = None
_model_ctc = None
model_embed = None
feature_extractor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Your provided Cell 3 Feedback Table
RULE_FEEDBACK = {
    "ikhfa": {"name": "Ikhfa (اخفاء)", "thresholds": {75: "Excellent Ikhfa", 40: "Good Ikhfa", 0: "Ikhfa Broken."}},
    "izhaar": {"name": "Izhaar (اظهار)", "thresholds": {75: "Excellent Izhaar", 60: "Good Izhaar", 40: "Izhaar Needs Work", 0: "Izhaar not clear."}},
    "qalqlah": {"name": "Qalqalah (قلقلة)", "thresholds": {75: "Excellent Qalqalah", 60: "Good Qalqalah", 0: "Qalqalah missing or weak."}},
     "mad": {"name": "mad (مد)", "thresholds": {75: "Excellent mad", 60: "Good mad", 0: "mad missing or weak."}},
     "mad asli": {"name": "mad asli (مد اصلی)", "thresholds": {75: "Excellent mad asli", 60: "Good mad asli", 0: "strech to one alif length."}},


}

def load_models():
    """Updates the module-level variables globally."""
    global _processor, _model_ctc, model_embed, feature_extractor
    
    MODEL_CTC = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    MODEL_BASE = "facebook/wav2vec2-base"
    
    _processor = Wav2Vec2Processor.from_pretrained(MODEL_CTC)
    _model_ctc = Wav2Vec2ForCTC.from_pretrained(MODEL_CTC).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_BASE)
    model_embed = Wav2Vec2Model.from_pretrained(MODEL_BASE).to(device)
    
    _model_ctc.eval()
    model_embed.eval()
    print("✅ Models loaded and assigned to audio_handler module.")

# --- Scroring Logic (Cell 2 & 3) ---
def get_embedding(audio_path, model, extractor, dev):

    y, _ = librosa.load(audio_path, sr=16000)
    inputs = extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(inputs.input_values.to(dev), output_hidden_states=True)
    averaged = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0).squeeze(0)
    frames = averaged.cpu().numpy()
    return frames.mean(axis=0), frames

def get_tajweed_verdict(rule_name, user_emb, ref_emb, user_frames, ref_frames):
    cos_sim = float(1 - cosine(ref_emb, user_emb))
    dist, _ = fastdtw(ref_frames, user_frames, dist=cosine)
    dtw_sim = max(0.0, 1.0 - (dist / (len(ref_frames) + len(user_frames))))
    score = ((cos_sim * 0.90) + (dtw_sim * 0.10)) * 100
    rule_key = rule_name.lower().strip().replace(" ", "_")
    rule = RULE_FEEDBACK.get(rule_key, {"thresholds": {0: "Keep Practicing"}})
    best_t = next((t for t in sorted(rule["thresholds"].keys(), reverse=True) if score >= t), 0)
    return rule["thresholds"][best_t]