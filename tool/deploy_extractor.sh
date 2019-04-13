#!/bin/bash

docker build -t ccr.ccs.tencentyun.com/prometheus/extractor-test-base:latest -f ./../docker-base/Dockerfile .

docker push ccr.ccs.tencentyun.com/prometheus/extractor-test-base:latest

sleep 5

kubectl --context=training-stage create -f ./../kube/extractor-pod.yaml
