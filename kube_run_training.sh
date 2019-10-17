#!/bin/bash

timestamp() {
  date +"%s"
}

# give a unique name to training pod. you could launch multiple pod at the same time now.
cat kube/training-pod.yamle | sed -e 's/pod-name/training-test-'$(timestamp)'/g' > kube/training-pod.yaml

# kubectl --context=training-prod -n monitoring delete pod training-test

docker build -t ccr.ccs.tencentyun.com/prometheus/training-test:latest -f docker-training/Dockerfile .
docker push ccr.ccs.tencentyun.com/prometheus/training-test:latest

sleep 60

# kubectl --context=training-stage create -f kube/training-pod.yaml
kubectl --context=training-prod create -f kube/training-pod.yaml
echo "training done"

