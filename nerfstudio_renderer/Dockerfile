FROM dromni/nerfstudio:0.3.4

ARG SERVER_PORT
ENV SERVER_PORT=$SERVER_PORT

# Copy nerfstudio_renderer source
COPY src/ /nerfstudio_renderer_source/

# pip-install nerfstudio_renderer
WORKDIR /nerfstudio_renderer_source
RUN sudo pip install .
WORKDIR /workspace

# Entrypoint: RPyC Server
ENTRYPOINT rpyc_classic --host 0.0.0.0 --port $SERVER_PORT
