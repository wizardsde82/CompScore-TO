FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir .
ENTRYPOINT ["compscore-train"]

