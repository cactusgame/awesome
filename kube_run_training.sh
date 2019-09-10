kubectl --context=training-prod -n monitoring delete pod training-test

sleep 120

docker build -t ccr.ccs.tencentyun.com/prometheus/training-test:latest -f docker-training/Dockerfile .
docker push ccr.ccs.tencentyun.com/prometheus/training-test:latest
# kubectl --context=training-stage create -f kube/training-pod.yaml
kubectl --context=training-prod create -f kube/training-pod.yaml
echo "training done"