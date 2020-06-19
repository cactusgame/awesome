I meet serval problems in this deploy, record here.  



1. config a wrong training branch.
   I planed to use the branch "feature_403" to deploy the new POE exp. So the config in abtest.xml likes that. 

   ```
   old algos:
   <variation algorithmId="poe_tv2_pb" percent="0" branch="release_1"/> 
   new algos: 
   <variation algorithmId="poe_tv3_pb" percent="0" branch="feature_403"/>  
   ```

   In fact, the training system doesn't support to train a same model-pipeline with different branch. So the model trained this early morning doesn't include the new POE experiment's signature. I find the problem in the canary test. Currently, you should config all the altos in a same branch. 
   

2. As I mentioned above, we need to "re-export" a model to do the deployment. 
   But I find that the "enable_warm_start" is turned off. The switch shouldn't be turned off. It's very dangerous to commit this kind of config when testing, a wrong model almost be deployed to production. 
   If you want to disable warm start in testing, I will make the variable can be passed outside the model.
   cc @rein @kevin
   

3. I find some of the new algos can't serve in production. The error is like that

   ```
   	 [[{{node Reshape_1208}}]]
   2020-02-21 11:26:46.799491: I tensorflow_serving/util/retrier.cc:33] Retrying of Loading servable: {name: model version: 1582252894} retry: 4
   2020-02-21 11:26:47.048592: I external/org_tensorflow/tensorflow/contrib/session_bundle/bundle_shim.cc:363] Attempting to load native SavedModelBundle in bundle-shim from: /opt/model/1582252894
   2020-02-21 11:26:47.048649: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: /opt/model/1582252894
   2020-02-21 11:26:47.102751: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
   2020-02-21 11:26:47.557752: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:182] Restoring SavedModel bundle.
   2020-02-21 11:26:48.077810: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:132] Running initialization op on SavedModel bundle.
   2020-02-21 11:26:48.433902: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:285] SavedModel load for tags { serve }; Status: success. Took 1385242 microseconds.
   2020-02-21 11:26:50.855533: E tensorflow_serving/util/retrier.cc:37] Loading servable: {name: model version: 1582252894} failed: Invalid argument: Input to reshape is a tensor with 0 values, but the requested shape has 1
   	 [[{{node Reshape_1208}}]]
   ```

   It means the "warm_up" failed. We suppose that the new user feature "pay_60days_14days_before" has been added to the targeting rule. But the "warm_up" file is random selected from the evaluation set. It doesn't contain the new feature "pay_60days_14days_before", so the model failed to load into TF-serving. 

   The stage testing didn't find the bug due to it uses the latest data which contains the feature "pay_60days_14days_before", but in production, it uses the recent 10 days red-log.

   In order to avoid the next POE training failed, I hotfix it in a hacky way. I override the "make_warm_up_file" in POE, all model use the new feature can't be warmed up. 

   ```
       def make_warm_up_file(self, path_local_request_data):
           return self._make_warm_up_file(path_local_request_data,
                                          ["poe_tv3_pb", "poe_tv1_pb", "poe_tv1_random", "poe_tv2_random", "poe_tv2_pb",
                                           "poe_tv2_alpha_1", "poe_tv2_alpha_2", "poe_tv2_alpha_3", "poe_tv2_s1d",
                                           "poe_tv2_s3d", "poe_tv2c_ratio_price_s3d_s1d", "poe_tv2c_pb_s1d",
                                           "poe_tv1_dec_random", "poe_nbv7_tv1_dec_p-b", "poe_tv2_price", "poe_nbv7"])
   ```

   I think if a new feature added to the model, we'd better to handle the situation that if the feature doesn't exist. 