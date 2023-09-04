# FROM nvidia/cuda:11.7.0-base-ubuntu22.04
FROM nvidia/cuda:11.7.1-base-ubuntu22.04

ARG work_dir="/work/"

WORKDIR ${work_dir}

RUN apt-get update && apt-get install -y \
    python3-pip \ 
    sudo \
    wget \
    git \
    vim \
    curl
RUN ln -s /usr/bin/python3.10 /usr/bin/python 
RUN pip install --upgrade pip
ENV POETRY_HOME=/usr/local
RUN curl -sSL https://install.python-poetry.org | python3 -
# シンボリックによるpathへのpoetryコマンドの追加
RUN ln -s /usr/local/bin/poetry /bin/poetry 
# 仮想環境を作成しない設定(コンテナ前提のため，仮想環境を作らない)
RUN poetry config virtualenvs.create false

COPY ./pyproject.toml  ./
RUN poetry install

# DDNM のクローン
RUN git clone https://github.com/wyhuai/DDNM.git

# デモのための画像の収納先作成
RUN  mkdir -p ./DDNM/exp/logs/imagenet ./DDNM/exp/logs/celeba

# デモのための画像入手
COPY ./celeba_hq.ckpt ./DDNM/exp/logs/celeba/celeba_hq.ckpt
RUN wget -P ./DDNM/exp/logs/imagenet https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

# stable diffusion のクローン
#RUN git clone https://github.com/Stability-AI/stablediffusion.git

# diffusers のクローン
RUN git clone https://github.com/huggingface/diffusers.git
#CMD ["/bin/bash"]




