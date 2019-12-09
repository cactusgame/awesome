#!/bin/bash

docker build -t ccr.ccs.tencentyun.com/prometheus/training-test:latest -f docker-training/Dockerfile .
docker push ccr.ccs.tencentyun.com/prometheus/training-test:latest

sleep 20

kubectl --context=training-prod create -f kube/testcase.yaml
echo "compile testcase done"

