kubectl --context=training-prod -n monitoring delete pod training-all-in-one

sleep 120

docker build -t ccr.ccs.tencentyun.com/prometheus/extractor-test:latest -f docker-extractor/Dockerfile .
docker push ccr.ccs.tencentyun.com/prometheus/extractor-test:latest
echo "extractor built"

docker build -t ccr.ccs.tencentyun.com/prometheus/training-test:latest -f docker-training/Dockerfile .
docker push ccr.ccs.tencentyun.com/prometheus/training-test:latest
echo "training built"

sleep 60

kubectl --context=training-prod create -f kube/all.yaml
echo "all pods done"