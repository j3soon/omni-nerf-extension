FROM dromni/nerfstudio:0.3.4

ARG SERVER_PORT
ENV SERVER_PORT=$SERVER_PORT

RUN sudo pip install rpyc

WORKDIR /workspace/src

# Force using the latest version of the source code for the ease of development
# Entrypoint: RPyC Server
ENTRYPOINT cp -r /workspace/src ~/src \
           && cd ~/src \
           && sudo pip install . \
           && python3 --version \
           && rpyc_classic --host 0.0.0.0 --port $SERVER_PORT
