#!/usr/bin/env bash
# Sets up the venv, authenticates, and runs Qualcomm's official export
# command for Llama-3.2-1B-Instruct. This is a thin wrapper only —
# it does not modify quantization flags, output handling, or auth beyond
# what qai_hub_models does itself.
set -e

# 1. Create + activate venv (skip if already created)
if [ ! -d "onnx-env" ]; then
    python -m venv onnx-env
fi
source onnx-env/Scripts/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Hugging Face auth (requires accepting Meta's Llama-3.2 license on
#    huggingface.co first, under your account)
echo "Logging in to Hugging Face (paste your access token when prompted)..."
hf auth login

# 4. Qualcomm AI Hub auth
read -p "Enter your Qualcomm AI Hub API token: " AI_HUB_TOKEN
qai-hub configure --api_token "$AI_HUB_TOKEN"

# 5. Official export command (compiles + quantizes to QNN for
#    Snapdragon X Elite). Inferencing/profiling are skipped here and
#    run separately in profile.py so NPU vs CPU numbers are captured
#    cleanly, matching the benchmark format used elsewhere in the repo.
python -m qai_hub_models.models.llama_v3_2_1b_instruct.export \
    --device "Snapdragon X Elite CRD" \
    --skip-inferencing \
    --skip-profiling \
    --output-dir ./export-output

echo "Export complete. Output in ./export-output"
echo "Next: run 'python profile.py' to benchmark NPU vs CPU."
