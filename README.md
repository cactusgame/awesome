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
kubectl --context=training-stage create -f kube/extractor-pod.yaml
kubectl --context=training-prod create -f kube/training-pod.yaml
echo "extracting done"
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

##### get model training run tfevents
```
coscmd -b peng-1256590953 download -rs models_training/rnn_v1/<model_version> /tmp/peng_model/

tensorboard --logdir=/tmp/peng_model/
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
kubectl --context=training-stage create -f kube/training-pod.yaml
kubectl --context=training-prod create -f kube/training-pod.yaml

echo "training done"

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

## Model Analysis
download the model for analysis  
```
coscmd -b peng-1256590953 download -r models_training/rnn_v2/1570120948 /tmp/peng_model/
```

launch tensorboard for analysis
```
tensorboard --logdir /tmp/peng_model/1567442811/
```



## Test result
#### prebuild_v1: LinearClassifier
features: share_id,close0-20
result  : {'loss': 11.054411, 'accuracy_baseline': 0.503875, 'global_step': 1000000, 'recall': 0.60543287, 'auc': 0.5334816, 'prediction/mean': 0.5041689, 'precision': 0.5238811, 'label/mean': 0.503875, 'average_loss': 0.6909007, 'auc_precision_recall': 0.53230953, 'accuracy': 0.5239375}

features: time,share_id,close0-20
result  :{'loss': 7.8904605, 'accuracy_baseline': 0.5008125, 'global_step': 1000000, 'recall': 0.740297, 'auc': 0.83925486, 'prediction/mean': 0.5027409, 'precision': 0.76670545, 'label/mean': 0.5008125, 'average_loss': 0.49315378, 'auc_precision_recall': 0.8436985, 'accuracy': 0.757125}
tip: I should not use feature `time`, because different shares in the same `time` has the same trend

#### prebuild_v2: DNNClassifier
features: share_id,close0-20
result: {'loss': 10.751576, 'accuracy_baseline': 0.5085625, 'global_step': 1000000, 'recall': 0.5825243, 'auc': 0.61233175, 'prediction/mean': 0.505836, 'precision': 0.5855466, 'label/mean': 0.5085625, 'average_loss': 0.6719735, 'auc_precision_recall': 0.62195647, 'accuracy': 0.578}

features: share_id,close0-20    
specific 2 classes and change the hidden layer         hidden_units = [256, 128, 64]

{'loss': 10.645561, 'accuracy_baseline': 0.5015, 'global_step': 1000000, 'recall': 0.6064307, 'auc': 0.63192403, 'prediction/mean': 0.5171307, 'precision': 0.58443433, 'label/mean': 0.5015, 'average_loss': 0.6653476, 'auc_precision_recall': 0.6384548, 'accuracy': 0.586375}



#### customized DNN
net: shared dense stack + dropout=0.5
INFO:tensorflow:Saving dict for global step 1000000: accuracy = 0.5325625, global_step = 1000000, loss = 0.6892759

net: (512,64) shared dense stack + dropout=0.2
{'loss': 0.71108305, 'global_step': 1000000, 'accuracy': 0.554125}

conclusion:
- it's easier to predict the next 1 day than the next 20 days when using the same feature(recent 20 days)