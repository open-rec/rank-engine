ARG RANK_BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
FROM ${RANK_BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_FIND_LINKS
ARG PIP_NO_INDEX
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    RANK_HOST=0.0.0.0 \
    RANK_PORT=8123

WORKDIR /opt/openrec/rank-engine
COPY requirements.txt requirements-common.txt requirements-cuda.txt ./
# Reuse the existing base image, then upgrade its Python runtime to cu129.
# Resolve Torch and nvForest together so both use the same CUDA libraries.
RUN python -c "import torch; assert torch.__version__.split('+')[0] == '2.10.0'; assert torch.version.cuda in ('12.8', '12.9')" \
    && pip uninstall -y torchaudio torchvision \
    && pip install --no-cache-dir -r requirements-cuda.txt \
    && pip check \
    && python -c "import torch, lightgbm, nvforest; assert torch.__version__ == '2.10.0+cu129'; assert torch.version.cuda == '12.9'"
COPY --from=algorithm . /tmp/rec-algorithm
RUN pip install --no-cache-dir /tmp/rec-algorithm && rm -rf /tmp/rec-algorithm
COPY . ./
RUN chmod +x start.sh

EXPOSE 8123
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8123/health', timeout=3)"

ENTRYPOINT ["./start.sh"]
