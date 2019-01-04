FROM ccr.ccs.tencentyun.com/prometheus/extractor-test-base:latest

# Base pip libaries
ADD docker-base/requirements.pip /root/requirements.pip
RUN pip --no-cache-dir install -r /root/requirements.pip && \
    python -m ipykernel.kernelspec

ADD src /root/src/