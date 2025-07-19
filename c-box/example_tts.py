import torch
import torchaudio as ta
from pydub import AudioSegment
from pathlib import Path
from src.tts import ChatterboxTTS
# from chatterbox.tts import ChatterboxTTS (in lib - not in src - old one)


# Detect device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Base directory where script is located
BASE_DIR = Path(__file__).resolve().parent

# === Audio Conversion Helper ===
def convert_to_wav(input_path: Path, sample_rate: int = 16000) -> Path:
    """Converts any audio format to mono .wav if needed"""
    if input_path.suffix.lower() == ".wav":
        return input_path  # No conversion needed
    
    output_path = input_path.with_suffix(".converted.wav")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(sample_rate).set_channels(1)
    audio.export(output_path, format="wav")
    print(f"🔁 Converted {input_path.name} -> {output_path.name}")
    return output_path

# Load TTS model
model = ChatterboxTTS.from_pretrained(device=device)

# Voice prompt (supports .m4a, .mp3, etc.)
voice_sample = BASE_DIR / "samples" / "shru.wav"
voice_sample = convert_to_wav(voice_sample)

# Text to synthesize
TEXT = (
    "Hi, good morning! Today I have completed working on the LLM integration and have 2 more tickets pending from the ASU board. "
    "I am expecting to complete them by end of this week. "
    "I do not have any blockers as of now and I will reach out to the team if required. "
    "Please feel free to reach out to me if you guys need anything. "
    "Thank you!"
)

# Generate audio using voice prompt
wav = model.generate(
    text=TEXT,
    audio_prompt_path=str(voice_sample),
)

# Save the output audio
output_path = BASE_DIR / "output" / "tts" / "op8.wav"
output_path.parent.mkdir(parents=True, exist_ok=True)
ta.save(str(output_path), wav.cpu(), model.sr)

print(f"✅ Done. Audio saved to {output_path}")
