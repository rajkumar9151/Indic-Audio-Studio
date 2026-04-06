#!/bin/bash
# Indic Audio Studio Setup Script

echo "--- Installing Python Dependencies ---"
pip install fastapi uvicorn pydantic transformers torch torchaudio soundfile

echo "--- Installing Parler-TTS from GitHub ---"
pip install git+https://github.com/huggingface/parler-tts.git

echo "--- Setup Complete ---"
echo "Run 'python main.py' to start the server."
