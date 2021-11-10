FROM python:3.7 as base

ARG USERNAME=coder
ARG USER_UID=1004
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

WORKDIR /home/${USERNAME}/src

COPY requirements.txt ./requirements.txt

RUN pip install -r ./requirements.txt

RUN spacy download en_core_web_md
