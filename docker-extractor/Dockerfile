FROM ccr.ccs.tencentyun.com/prometheus/extractor-test-base:latest

# Base pip libaries
ADD docker-base/requirements.pip /root/requirements.pip
RUN pip --no-cache-dir install -r /root/requirements.pip && \
    python -m ipykernel.kernelspec

ADD docker-extractor/.cos.conf /root/.cos.conf
ADD src /root/src/
ADD run_extractor.sh /root/run_extractor.sh