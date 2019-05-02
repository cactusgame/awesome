# awesome

## Feature Extractor
the extractor is only responsible for fetching data through internet.  
If you want to use some of the features in your algorithm, you should implement a `Data Formatter` to define which features you will use.
  
### Install
#### the base image for feature extractor. 
You must exec this command under the root (awesome) folder
This part depends on Tencent's cloud
```
(docker login...)   
docker build -t ccr.ccs.tencentyun.com/prometheus/extractor-test-base:latest -f docker-base/Dockerfile .
docker push ccr.ccs.tencentyun.com/prometheus/extractor-test-base:latest
```

#### package the feature extractor image
```
docker build -t ccr.ccs.tencentyun.com/prometheus/extractor-test:latest -f docker-extractor/Dockerfile .
docker push ccr.ccs.tencentyun.com/prometheus/extractor-test:latest
```

#### run the feature extractor by Pod
You have to push the docker image first, then create it in k8s.
```
kubectl --context=training-stage create -f kube/extractor-pod.yaml
```

#### run the feature extractor by Pod locally
```
docker run -it ccr.ccs.tencentyun.com/prometheus/extractor-test:latest bash
```

##### export FEATURES in sqlite DB to .csv manually
enter into sqlite (in project `root` folder)
```
sqlite3 awesome.db

.exit (exit the console)
```
generate features.csv in the `data` folder
```
.header on  
.mode csv  
.once ./data/features.csv
SELECT * FROM FEATURE;
```


## Algorithm training pipeline
The pipeline is split into several parts
- preprocessor : download the feature db from COS. Then convert the features defined in `DataFormatter` to `TFRecord`
- trainer: use a specific algorithm to train the model

#### run training in docker
```
docker build -t ccr.ccs.tencentyun.com/prometheus/training-test:latest -f docker-training/Dockerfile .
docker push ccr.ccs.tencentyun.com/prometheus/training-test:latest

docker run -it ccr.ccs.tencentyun.com/prometheus/training-test:latest bash
```
#### run training locally
```
python src/algorithm/xxx_algorithm/main.py
```

#### run the feature extractor by Pod
You have to push the docker image first, then create it in k8s.
```
kubectl --context=training-stage create -f kube/training-pod.yaml
```

## TODO list
- 有关熵的理解
- end to end sample
- tensor board
- model anaylis
- add `share_id`, `time` features