FROM mambaorg/micromamba:1.4.7

USER root
# Keep the base environment activated
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN apt update && apt -y install curl git gcc g++ graphviz make libxrender1

COPY requirements.txt  /tmp/requirements.txt
WORKDIR /tmp

RUN micromamba install -y python=3.10.12 pip=23.2.1 -c conda-forge
RUN pip install setuptools==81
RUN pip install -r requirements.txt

COPY . /canonicalization-server
WORKDIR /canonicalization-server

EXPOSE 3319

CMD ["python", "canonicalization_server.py"]
