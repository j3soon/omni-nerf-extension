FROM python:3.10

RUN apt-get update \
    && apt-get install -y libgl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /root
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /workspace

ENTRYPOINT bash
