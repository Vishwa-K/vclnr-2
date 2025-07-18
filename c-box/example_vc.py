import torch
import torchaudio as ta
from chatterbox.vc import ChatterboxVC  # ✅ Use TTS model instead of VC

# Detect device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# Load TTS model
model = ChatterboxVC.from_pretrained(device=device)


voice_sample = "samples/ap1.wav"  # Reference voice sample (your voice)
target_sample = "samples/my_voice3.wav"
# Generate speech in the reference voice
wav = model.generate(
    audio = voice_sample,
    target_voice_path=target_sample,
)

# Save the audio
ta.save("output/vc/apTigervc2.wav", wav.cpu(), model.sr)
