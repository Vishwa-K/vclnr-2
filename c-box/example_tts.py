import torch
import torchaudio as ta
from pathlib import Path
from chatterbox.tts import ChatterboxTTS 

# Detect device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Base directory where script is located
BASE_DIR = Path(__file__).resolve().parent

# Load TTS model
model = ChatterboxTTS.from_pretrained(device=device)

# Voice prompt (your voice sample)
voice_sample = BASE_DIR / "samples" / "ap1.wav"  

# Text to synthesize
TEXT = "Hi, good morning! Today I have completed working on the LLM integration and have 2 more tickets pending from the ASU board."
"I am expecting to complete them by end of this week,"
"I do not have any blockers as of now and I will reach out to the team if required."
"Please feel free to reach out to me know if you guys need anything"
"Thank you!"

# Generate audio using voice prompt
wav = model.generate(
    text=TEXT,
    audio_prompt_path=str(voice_sample),  # must be string
)

# Save the output audio
output_path = BASE_DIR / "output" / "tts" / "op5.wav"
output_path.parent.mkdir(parents=True, exist_ok=True)  # Make sure directory exists
ta.save(str(output_path), wav.cpu(), model.sr)

print(f"✅ Done. Audio saved to {output_path}")
