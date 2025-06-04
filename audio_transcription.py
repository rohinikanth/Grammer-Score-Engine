# -*- coding: utf-8 -*-
!pip install openai-whisper

!git clone https://github.com/rohinikanth/Grammer-Score-Engine.git

!pip install openai-whisper
#Testing with an example
'''
import whisper
model = whisper.load_model("medium")
#Example usage
result = model.transcribe("/content/Grammer-Score-Engine/Dataset/audios/train/audio_399.wav")
text = result["text"]

print("Transcribed Text:", text)
'''

import os
import re
import unicodedata
import torch
import whisper
import pandas as pd

# -------------------------------------------------------------------
# Force PyTorch to use FP32 by default (silences the FP16 warning)
# -------------------------------------------------------------------
torch.set_default_dtype(torch.float32)

# -------------------------------------------------------------------
# Transcript cleaning function
# -------------------------------------------------------------------
def clean_transcript(text: str) -> str:
    """
    Normalize and clean Whisper output:
    - remove non-ASCII / IPA symbols
    - collapse long repeats of chars, words, punctuation
    - drop anything <3 words or lacking real words
    """
    if not isinstance(text, str):
        return ""
    # Unicode normalize
    text = unicodedata.normalize("NFKC", text).lower().strip()
    # Remove non-ASCII
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Collapse repeated chars (e.g., "rrrrrrrr")
    text = re.sub(r"(.)\1{4,}", r"\1", text)
    # Remove long punctuation runs
    text = re.sub(r"[\.,;!?'\-]{3,}", " ", text)
    # Collapse repeated words ("the the the" -> "the")
    text = re.sub(r"\b(\S+)( \1\b)+", r"\1", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Drop if too short or no real words
    if len(text.split()) < 3 or not re.search(r"\w{3,}", text):
        return ""
    return text

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
AUDIO_DIR = "/content/Grammer-Score-Engine/Dataset/audios/train"
CSV_PATH   = "/content/ProsodicFeatures.csv"
LOG_PATH   = "transcription_progress.log"
WHISPER_SIZE = "small"
# -------------------------------------------------------------------
# Load model & data
# -------------------------------------------------------------------
model = whisper.load_model(WHISPER_SIZE)
df = pd.read_csv(CSV_PATH)

# Ensure a 'transcript' column exists
if "transcript" not in df.columns:
    df["transcript"] = ""

# -------------------------------------------------------------------
# Transcription loop
# -------------------------------------------------------------------
for idx, row in df.iterrows():
    fname = row["file_name"]
    path  = os.path.join(AUDIO_DIR, fname)

    # Skip if we already have a decent transcript
    existing = str(row.get("transcript", "")).strip()
    if existing and len(existing.split()) > 2:
        continue

    if not os.path.isfile(path):
        msg = f"[{idx+1}/{len(df)}] File not found: {fname}"
        df.at[idx, "transcript"] = ""
    else:
        try:
            res = model.transcribe(path, language="en", fp16=False)
            raw = res.get("text", "")
            clean = clean_transcript(raw)
            df.at[idx, "transcript"] = clean
            if clean:
                msg = f"[{idx+1}/{len(df)}] ✔ Transcribed: {fname}"
            else:
                msg = f"[{idx+1}/{len(df)}] ✘ Skipped (no valid transcript): {fname}"
        except Exception as e:
            df.at[idx, "transcript"] = ""
            msg = f"[{idx+1}/{len(df)}] ⚠ Error {fname}: {e}"

    # Log & print
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(msg + "\n")
    print(msg)

    # Save checkpoint every 10
    if (idx + 1) % 10 == 0:
        df.to_csv(CSV_PATH, index=False)
        print(f"--- Checkpoint saved at row {idx+1} ---")

# Final save
df.to_csv(CSV_PATH, index=False)
print(" All done. CSV updated with transcripts.")



