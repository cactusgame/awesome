docker build -t ccr.ccs.tencentyun.com/prometheus/extractor-test:latest -f docker-extractor/Dockerfile .
docker push ccr.ccs.tencentyun.com/prometheus/extractor-test:latest
kubectl --context=training-stage create -f kube/extractor-pod.yaml
echo "extracting done"