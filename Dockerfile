FROM continuumio/anaconda3:2019.10

LABEL version="0.1"
LABEL maintainer="diracdiego@gmail.com"

SHELL ["/bin/bash", "-c"]

# Create base conda environment
ADD ./env.yml /
RUN conda env create -n work -f /env.yml && rm -rf /env.yml

RUN conda init bash
RUN source ~/.bashrc

WORKDIR /work

ENTRYPOINT ["/bin/bash"]