import os
import datetime
import torch
import torchaudio

# Force cache directories to point to the local models/ directory
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")
os.environ["HF_HOME"] = models_dir
os.environ["TORCH_HOME"] = models_dir

from stable_audio_3 import StableAudioModel

def main():
    # Parse prompt.txt
    prompt_file = os.path.join(current_dir, "prompt.txt")
    duration = None
    prompt_text = None

    with open(prompt_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    for i, line in enumerate(lines):
        if line.strip() == "[DURATION]" and i + 1 < len(lines):
            duration = int(lines[i+1].strip())
        elif line.strip() == "[PROMPT]" and i + 1 < len(lines):
            prompt_text = lines[i+1].strip()

    if duration is None or prompt_text is None:
        raise ValueError("Could not parse [DURATION] or [PROMPT] from prompt.txt")

    # Initialize the model onto CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableAudioModel.from_pretrained("medium").to(device)

    # Synthesize the audio
    # Assuming model.generate returns a tuple (audio_tensor, sample_rate) or an object
    output = model.generate(prompt_text, duration)

    # Extract the audio tensor and sample rate
    if isinstance(output, tuple) and len(output) >= 2:
        audio_tensor, sample_rate = output[0], output[1]
    else:
        # Fallback if output is an object with attributes
        audio_tensor = output.audio_tensor if hasattr(output, 'audio_tensor') else output.audio
        sample_rate = output.sample_rate

    # Ensure tensor is on CPU before saving
    audio_tensor = audio_tensor.cpu()

    # Save the resulting .wav file into the output/ folder
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"audio_{timestamp}.wav")

    torchaudio.save(output_path, audio_tensor, sample_rate)
    print(f"Successfully generated and saved to {output_path}")

if __name__ == "__main__":
    main()
