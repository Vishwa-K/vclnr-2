from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import scipy.signal

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond

REPO_ID = "ResembleAI/chatterbox"

def punc_norm(text: str) -> str:
    if len(text) == 0:
        return "You need to add some text for me to talk."

    if text[0].islower():
        text = text[0].upper() + text[1:]

    text = " ".join(text.split())

    punc_to_replace = [
        ("...", ", ..."),
        ("…", ", ..."),
        (":", ","),
        (" - ", ","),
        (";", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}):
        text += "."

    return text

@dataclass
class Conditionals:
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3.speaker_emb = self.t3.speaker_emb.to(device)
        if self.t3.cond_prompt_speech_tokens is not None:
            self.t3.cond_prompt_speech_tokens = self.t3.cond_prompt_speech_tokens.to(device)
        self.t3.emotion_adv = self.t3.emotion_adv.to(device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(t3=self.t3.__dict__, gen=self.gen)
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])

class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, t3, s3gen, ve, tokenizer, device, conds=None):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device):
        ckpt_dir = Path(ckpt_dir)
        map_location = torch.device('cpu') if device in ["cpu", "mps"] else None

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        t3.load_state_dict(t3_state.get("model", [t3_state])[0])
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        conds = None
        if (ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(ckpt_dir / "conds.pt", map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device):
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(hf_hub_download(repo_id=REPO_ID, filename="conds.pt")).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.85):
        s3gen_ref_wav, _ = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)

        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)).mean(axis=0, keepdim=True)
        emotion_level = torch.tensor([[[exaggeration]]]).clamp(0.6, 1.2)

        t3_cond = T3Cond(
            speaker_emb=ve_embed.to(self.device),
            cond_prompt_speech_tokens=t3_cond_prompt_tokens.to(self.device) if t3_cond_prompt_tokens is not None else None,
            emotion_adv=emotion_level.to(self.device),
        )

        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(self, text, repetition_penalty=1.1, min_p=0.05, top_p=0.85,
                 audio_prompt_path=None, exaggeration=0.85, cfg_weight=0.6, temperature=1.0):

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please prepare_conditionals first or provide audio_prompt_path."

        _cond: T3Cond = self.conds.t3
        self.conds.t3 = T3Cond(
            speaker_emb=_cond.speaker_emb.to(self.device),
            cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens.to(self.device) if _cond.cond_prompt_speech_tokens is not None else None,
            emotion_adv=torch.tensor([[[exaggeration]]]).clamp(0.6, 1.2).to(self.device),
        )

        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        text_tokens = F.pad(text_tokens, (1, 0), value=self.t3.hp.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=self.t3.hp.stop_text_token)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )[0]

            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens[speech_tokens < 6561].to(self.device)

            wav, _ = self.s3gen.inference(speech_tokens=speech_tokens, ref_dict=self.conds.gen)
            wav = wav.squeeze(0).detach().cpu().numpy()

            b, a = scipy.signal.butter(1, 100 / (self.sr / 2), btype='highpass')
            wav = scipy.signal.filtfilt(b, a, wav)

            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)

        return torch.from_numpy(watermarked_wav).unsqueeze(0)
