
kubectl --context=training-prod -n monitoring delete pod extractor-test

sleep 120

docker build -t ccr.ccs.tencentyun.com/prometheus/extractor-test:latest -f docker-extractor/Dockerfile .
docker push ccr.ccs.tencentyun.com/prometheus/extractor-test:latest

sleep 60

# kubectl --context=training-stage create -f kube/extractor-pod.yaml
kubectl --context=training-prod create -f kube/extractor-pod.yaml

echo "extracting done"