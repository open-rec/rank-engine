ARG RANK_BASE_IMAGE=mirrors-ssl.aliyuncs.com/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
FROM ${RANK_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    RANK_HOST=0.0.0.0 \
    RANK_PORT=8000

WORKDIR /opt/openrec/rank-engine
COPY requirements-common.txt ./
RUN pip install --no-cache-dir -r requirements-common.txt
COPY --from=algorithm . /tmp/rec-algorithm
RUN pip install --no-cache-dir /tmp/rec-algorithm && rm -rf /tmp/rec-algorithm
COPY . ./
RUN chmod +x start.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); assert d['code']==0"

ENTRYPOINT ["./start.sh"]
