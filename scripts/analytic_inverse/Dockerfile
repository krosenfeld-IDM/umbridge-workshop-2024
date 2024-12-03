FROM ubuntu:latest

# Define the build argument
ARG MODEL
ENV model=${MODEL}

COPY . /server

RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y python3-pip python3-venv && \
    python3 -m venv venv && \
    . venv/bin/activate && \
    pip install umbridge numpy

# Use the build argument in the CMD instruction
CMD /venv/bin/python /server/server_${model}.py
