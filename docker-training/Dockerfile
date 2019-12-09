FROM ccr.ccs.tencentyun.com/prometheus/extractor-test-base:latest

# Base pip libaries
ADD docker-training/requirements.pip /root/requirements.pip
RUN pip --no-cache-dir install -r /root/requirements.pip && \
    python -m ipykernel.kernelspec

ADD docker-training/.cos.conf /root/.cos.conf
ADD run_training.sh /root/run_training.sh
ADD run_testcase.sh /root/run_testcase.sh
ADD src /root/src/
