FROM nvcr.io/nvidia/isaac-sim:2023.1.1
# Ref: https://stackoverflow.com/a/53361581
RUN apt-get update \
    && apt-get install -y libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN /isaac-sim/python.sh -m pip install opencv-python rpyc
