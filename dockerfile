FROM python:3.7 as base

ARG USERNAME=coder
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

WORKDIR /home/${USERNAME}/src

COPY requirements.txt ./requirements.txt

RUN pip install -r ./requirements.txt --index-url http://nexus.prod.uci.cu/repository/pypi-proxy/simple/ --trusted-host nexus.prod.uci.cu

FROM base as prod

RUN spacy download en

COPY --chown=${USERNAME}:${USER_GID} . .
