import os
import torch
import soundfile as sf
from datetime import datetime

# ==============================================================================
# 1. DIRECTORY CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
T5_DIR = os.path.join(MODELS_DIR, "t5gemma-b-b-ul2")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
PROMPT_FILE = os.path.join(SCRIPT_DIR, "prompt.txt")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lock cache behavior
os.environ["HF_HOME"] = MODELS_DIR
os.environ["TORCH_HOME"] = MODELS_DIR
os.environ["HF_HUB_OFFLINE"] = "1"

# ==============================================================================
# 2. LOCAL FILE INTERCEPTORS
# ==============================================================================
import huggingface_hub
_orig_hf_download = huggingface_hub.hf_hub_download

def _mock_hf_download(repo_id, filename, *args, **kwargs):
    subfolder = kwargs.get("subfolder", "")
    candidates = [
        os.path.join(MODELS_DIR, filename),
        os.path.join(MODELS_DIR, subfolder, filename) if subfolder else None,
        os.path.join(T5_DIR, filename),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return _orig_hf_download(repo_id, filename, *args, **kwargs)

huggingface_hub.hf_hub_download = _mock_hf_download

import transformers

def _resolve_local_path(pretrained_path, kwargs):
    subfolder = kwargs.pop("subfolder", None)
    if subfolder == "t5gemma-b-b-ul2" or (isinstance(pretrained_path, str) and ("t5gemma" in pretrained_path or "stable-audio-3" in pretrained_path)):
        if os.path.exists(T5_DIR):
            return T5_DIR
    return pretrained_path

_orig_auto_tok = transformers.AutoTokenizer.from_pretrained
_orig_auto_cfg = transformers.AutoConfig.from_pretrained
_orig_pt_model = transformers.PreTrainedModel.from_pretrained

@classmethod
def _mock_auto_tok(cls, pretrained_model_name_or_path, *args, **kwargs):
    target = _resolve_local_path(pretrained_model_name_or_path, kwargs)
    return _orig_auto_tok.__func__(cls, target, *args, **kwargs)

@classmethod
def _mock_auto_cfg(cls, pretrained_model_name_or_path, *args, **kwargs):
    target = _resolve_local_path(pretrained_model_name_or_path, kwargs)
    return _orig_auto_cfg.__func__(cls, target, *args, **kwargs)

@classmethod
def _mock_pt_model(cls, pretrained_model_name_or_path, *args, **kwargs):
    target = _resolve_local_path(pretrained_model_name_or_path, kwargs)
    return _orig_pt_model.__func__(cls, target, *args, **kwargs)

transformers.AutoTokenizer.from_pretrained = _mock_auto_tok
transformers.AutoConfig.from_pretrained = _mock_auto_cfg
transformers.PreTrainedModel.from_pretrained = _mock_pt_model

# ==============================================================================
# 3. PROMPT PARSER
# ==============================================================================
from stable_audio_3 import StableAudioModel

def parse_prompt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing prompt file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    duration = 30
    prompt_lines = []
    current_section = None

    for line in content.splitlines():
        line_clean = line.strip()
        if line_clean.startswith("[") and line_clean.endswith("]"):
            current_section = line_clean[1:-1].upper()
        elif line_clean:
            if current_section == "DURATION":
                try:
                    duration = int(line_clean)
                except ValueError:
                    duration = 30
            elif current_section == "PROMPT":
                prompt_lines.append(line_clean)

    return duration, " ".join(prompt_lines)

# ==============================================================================
# 4. INFERENCE PIPELINE
# ==============================================================================
def main():
    print("=" * 60)
    print(" Stable Audio 3 Standalone Inference Engine")
    print("=" * 60)

    duration, prompt_text = parse_prompt(PROMPT_FILE)
    print(f"[Directing] Duration : {duration}s")
    print(f"[Directing] Prompt   : {prompt_text}")
    print("-" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Loader] Loading local model weights onto {device.upper()}...")

    model = StableAudioModel.from_pretrained("medium", device=device)

    print("[Pipeline] Synthesizing audio...")
    audio = model.generate(
        prompt=prompt_text,
        duration=duration
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_file = os.path.join(OUTPUT_DIR, f"stable_{timestamp}.wav")

    # Extract waveform and sample rate
    if isinstance(audio, tuple):
        wav, sample_rate = audio
    else:
        wav = audio
        sample_rate = getattr(model, "sample_rate", 44100)

    if isinstance(wav, torch.Tensor):
        audio_np = wav.detach().cpu().numpy()
    else:
        audio_np = wav

    # Squeeze extra batch dimensions and format channels for soundfile (samples, channels)
    if audio_np.ndim == 3:
        audio_np = audio_np.squeeze(0)
    if audio_np.ndim == 2 and audio_np.shape[0] < audio_np.shape[1]:
        audio_np = audio_np.T

    sf.write(out_file, audio_np, sample_rate)

    print("-" * 60)
    print(f"[SUCCESS] Audio generated and saved to:")
    print(f"          -> {out_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()