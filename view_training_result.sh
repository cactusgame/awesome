#!/bin/bash
# example: sh view_training_result.sh dnn_v1/1570970280

echo "Positional Parameters. You should pass $1 as {model_name}/{model_version}"
echo '$0 = ' $0
echo '$1 = ' $1

# remove folder /tmp/peng_model
rm -rf /tmp/peng_model/

# download analysis file from COS
coscmd -b peng-1256590953 download -r models_training/$1 /tmp/peng_model/

# launch tensorboard
tensorboard --logdir=/tmp/peng_model