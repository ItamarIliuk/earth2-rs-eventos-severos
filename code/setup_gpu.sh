#!/usr/bin/env bash
# Ambiente para rodar o ensemble FCN3 numa GPU de 80 GB (testado numa H100 PCIe, Ubuntu 24.04, driver com CUDA 12.8).
# Rodar de dentro da pasta code/:  bash setup_gpu.sh
#
# Receita reconstruída a partir da instalação que funcionou; os passos que deram problema estão comentados.
set -euo pipefail
cd "$(dirname "$0")"

sudo apt-get update -y
sudo apt-get install -y git curl cmake python3-dev libeccodes-dev ffmpeg

# nvcc é necessário para o torch-harmonics compilar a extensão CUDA. Sem ela, a operação DISCO do FCN3
# cai para a CPU sem aviso e cada passo fica dezenas de vezes mais lento.
# Precisa do repositório apt da NVIDIA. Se o apt reclamar de "Conflicting values set for option Signed-By",
# há duas entradas para o mesmo repositório: remova a do cuda-keyring e fique com a da imagem.
if [ ! -x /usr/local/cuda-12.6/bin/nvcc ]; then
  sudo apt-get install -y cuda-toolkit-12-6
fi
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export FORCE_CUDA_EXTENSION=1
export TORCH_CUDA_ARCH_LIST="9.0"   # H100; troque para a sua GPU (A100 = 8.0)

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# torch antes de tudo: torch-harmonics compila sem isolamento e precisa encontrá-lo
uv venv --python 3.13
uv pip install "torch==2.14.1" "torchvision==0.29.1" --index-url https://download.pytorch.org/whl/cu126
uv sync

uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
uv run python -c "from earth2studio.models.px import FCN3; print('FCN3 ok')"
find .venv -path "*torch_harmonics*" -name "*cuda*.so" | head -3
echo SETUP_OK
