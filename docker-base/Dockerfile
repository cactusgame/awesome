FROM ubuntu:16.04

# Base libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        libsnappy-dev \
        pkg-config \
        python-dev \
        rsync \
        software-properties-common \
        ssh-client \
        unzip \
        vim \
        zip \
        zlib1g-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install tccli tensorflow==1.8.0 tensorboard==1.8.0 coscmd==1.8.5.21


WORKDIR /root

