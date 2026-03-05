from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from contextlib import asynccontextmanager
import os, shutil, librosa, pandas as pd

# --- IMPORT THE MODULE DIRECTLY ---
import utils.audio_handler as ah
from utils.alignment import forced_align
from utils.phoneme_analysis import get_full_analysis

import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, ah.load_models)
    yield

app = FastAPI(lifespan=lifespan)
df = pd.read_excel("data/metadata.xlsm", engine="openpyxl")
df.columns = df.columns.str.lower()

@app.post("/upload-audio")
async def analyze_tajweed(word_id: str = Query(...), file: UploadFile = File(...)):
    # Check the live module variable
    if ah._processor is None:
        raise HTTPException(status_code=503, detail="Models still loading...")

    temp_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        row = df[df["id"] == word_id.strip()].iloc[0]
        audio_len = librosa.get_duration(path=temp_path)
        pred_ids, frames, detected_list = forced_align(temp_path)

        # Use the processor from the 'ah' module
        accuracy, phonetic_fb, expected, detected = get_full_analysis(
            row["phenome"], pred_ids, frames, audio_len, ah._processor
        )

        # Use model_embed and extractor from 'ah' module
        ref_path = f"data/All_Audio/{word_id.strip()}.mp3"
        u_emb, u_frames = ah.get_embedding(temp_path, ah.model_embed, ah.feature_extractor, ah.device)
        r_emb, r_frames = ah.get_embedding(ref_path, ah.model_embed, ah.feature_extractor, ah.device)
        
        tajweed_verdict = ah.get_tajweed_verdict(row["rule"], u_emb, r_emb, u_frames, r_frames)

        return {
            "Accuracy": f"{accuracy}%",
            "Phonetic Feedback": "\n".join(phonetic_fb) if phonetic_fb else "Perfect Pronunciation.",
            "Tajweed Feedback": tajweed_verdict,
            "Target": " ".join(expected),
            "You said": " ".join(detected)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)


    if __name__ == "__main__":
     import uvicorn
     port = int(os.environ.get("PORT", 8000))
     uvicorn.run(app, host="0.0.0.0", port=port)   