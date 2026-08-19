import os
import time
import torch
import torchaudio

# Force cache directories (HF_HOME and TORCH_HOME) to point to the local models/ directory
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "models")
os.environ["HF_HOME"] = models_dir
os.environ["TORCH_HOME"] = models_dir

from stable_audio_3 import StableAudioModel

def parse_prompt(filepath):
    duration = None
    prompt = None

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Simple parser for the specified format
    lines = content.strip().split('\n')

    duration_idx = -1
    prompt_idx = -1

    for i, line in enumerate(lines):
        if line.strip() == "[DURATION]":
            duration_idx = i
        elif line.strip() == "[PROMPT]":
            prompt_idx = i

    if duration_idx != -1 and duration_idx + 1 < len(lines):
        try:
            duration = int(lines[duration_idx + 1].strip())
        except ValueError:
            duration = None

    if prompt_idx != -1 and prompt_idx + 1 < len(lines):
        # The prompt might be multiple lines, so we join everything after [PROMPT]
        prompt = "\n".join([lines[i] for i in range(prompt_idx + 1, len(lines))]).strip()

    return duration, prompt

def main():
    prompt_file = os.path.join(base_dir, "prompt.txt")
    duration, prompt = parse_prompt(prompt_file)

    if duration is None or prompt is None:
        raise ValueError("Could not parse [DURATION] and [PROMPT] from prompt.txt")

    print(f"Duration: {duration}")
    print(f"Prompt: {prompt}")

    # Initialize the model onto CUDA
    print("Loading model...")
    model = StableAudioModel.from_pretrained("medium")
    model = model.to("cuda")

    # Synthesize the audio
    print("Generating audio...")
    output = model.generate(prompt=prompt, duration=duration)

    # Extract the audio tensor and sample rate
    if isinstance(output, tuple):
        audio_tensor, sample_rate = output[0], output[1]
    elif isinstance(output, dict):
        audio_tensor = output.get("audio", output.get("waveform"))
        sample_rate = output.get("sample_rate", getattr(model, "sample_rate", 44100))
    else:
        audio_tensor = output
        sample_rate = getattr(model, "sample_rate", 44100)

    # Save the resulting .wav file into the output/ folder with a timestamped filename
    output_dir = os.path.join(base_dir, "output")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"audio_{timestamp}.wav"
    output_path = os.path.join(output_dir, filename)

    print(f"Saving audio to {output_path}...")
    # Ensure tensor is on CPU and has correct dimensions for saving
    audio_tensor_cpu = audio_tensor.cpu()
    if audio_tensor_cpu.dim() == 1:
        audio_tensor_cpu = audio_tensor_cpu.unsqueeze(0)

    torchaudio.save(output_path, audio_tensor_cpu, sample_rate)
    print("Done.")

if __name__ == "__main__":
    main()
