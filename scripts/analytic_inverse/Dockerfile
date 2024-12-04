FROM ubuntu:latest

# Define the build argument
ARG MODEL
ENV model=${MODEL}

COPY . /server

RUN apt update && \
    apt install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \ 
    . $HOME/.local/bin/env && \
    uv venv --python=3.10 && \
    . .venv/bin/activate && \
    uv pip install umbridge numpy pydantic

# Use the build argument in the CMD instruction
CMD .venv/bin/python /server/server_${model}.py
