FROM mambaorg/micromamba:1.4.7

USER root
# Keep the base environment activated
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN apt update && apt -y install git gcc g++ make

COPY . /app/PolymerScribe
WORKDIR /app/PolymerScribe

RUN micromamba install -y python=3.10.12 pip=23.2.1 -c conda-forge
RUN pip install -r requirements.txt

ENV CUDA_VISIBLE_DEVICES 10

EXPOSE 3310

CMD ["python", "polymerscribe_server.py"]
