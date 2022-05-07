FROM ubuntu:20.04
USER root
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV DEBIAN_FRONTEND noninteractive

RUN rm -rf /var/lib/apt/lists/* && apt-get update -y
RUN apt-get install -y curl \ 
    build-essential \
    vim \
    git \
    python3 \
    python3-dev \
    python3-pip \
    && echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

ENV LAN ja_JP.UTF-8
ENV TZ JST-9

COPY ./inference_server/requirements.txt /tmp/

RUN pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install -r /tmp/requirements.txt

# 現状はCPU版
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN git clone https://github.com/pytorch/fairseq && \
    cd fairseq && \
    pip install --editable ./ && \
    export PYTHONPATH="/fairseq:$PYTHONPATH"

RUN mkdir /app/
