import os
import datetime
import torchaudio
from stable_audio_3 import StableAudioModel

def parse_prompt(file_path):
    duration = 0
    prompt = ""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    duration_idx = -1
    prompt_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "[DURATION]":
            duration_idx = i
        elif line.strip() == "[PROMPT]":
            prompt_idx = i

    if duration_idx != -1 and duration_idx + 1 < len(lines):
        duration = int(lines[duration_idx + 1].strip())

    if prompt_idx != -1 and prompt_idx + 1 < len(lines):
        prompt = lines[prompt_idx + 1].strip()

    return duration, prompt

def main():
    # Force cache directories to point to local models/ directory
    local_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(local_models_dir, exist_ok=True)
    os.environ["HF_HOME"] = local_models_dir
    os.environ["TORCH_HOME"] = local_models_dir

    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.txt")
    duration, prompt = parse_prompt(prompt_path)

    print(f"Parsed Duration: {duration}")
    print(f"Parsed Prompt: {prompt}")

    # Initialize model onto CUDA
    print("Initializing model...")
    model = StableAudioModel.from_pretrained("medium")
    model = model.to("cuda")

    # Generate audio
    print("Generating audio...")
    audio, sample_rate = model.generate(prompt=prompt, duration=duration)

    # Save the resulting .wav file
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"output_{timestamp}.wav")

    torchaudio.save(output_path, audio, sample_rate)
    print(f"Audio saved to {output_path}")

if __name__ == "__main__":
    main()
