## 背景

AUC（Area Under Curve）是机器学习中常用的一个评价指标，但是，想彻底理解这个指标并不是那么容易，本文希望通过一个例子彻底说明AUC到底是什么。

## 常见指标

在使用tensorflow的内建的"分类"模型完成一次训练及验证时，通常会有如下输出

```
 'loss': 11.080771,
 'accuracy_baseline': 0.5389375,
 'global_step': 2000,
 'recall': 0.8013452,
 'auc': 0.50163,
 'prediction/mean': 0.5355787,
 'precision': 0.53824586,
 'label/mean': 0.5389375,
 'average_loss': 0.6925482,
 'auc_precision_recall': 0.53178287,
 'accuracy': 0.5224375
```

上面包含了若干指标，用于评价该"分类器"的好坏，简要描述一下这些指标的含义。

### label/mean 

这是个二分类问题的输出，正例（positive）标记为1，负例（negative）标记为0，label/mean的意义就是验证集中所有标记的平均数。这里的`0.5389375`说明验证集中，正例略多一些，所以平均数大于0.5。类似的，`prediction/mean`的意思就是模型预测的结果的平均数，也是略倾向于正例的。

### accuracy
刚学习机器学习时，总是分不清什么准确度，精度等概念。后来我终于想明白了，混淆这些概念的根本原因是中文里准确度，精确度，精准度往往表达的是一个意思。所以为了不混淆，这些指标还是尽量用英文表示。

`accuracy`是最直观，最常用的指标，说白了，就是该模型有多少比例的结果预测对了，无论预测对了正例（TP）还是反例（TN）。accuracy是一个简单指标，不是什么时候都适用：

- 如果模型用于辨识性别，由于世界上男女数量大致相等，如果accuracy高，说明这个模型是可以正确分辨性别的。
- 如果模型用于辨识游戏中的作弊用户。假设1万名玩家中，有1位是作弊用户（TP），即便模型什么都没有学习到，模型只用回答所有玩家都不是作弊玩家，那么accuracy依然高达9999/10000=99.99%。

由此可见，如果分类A和分类B的数量非常不平衡时，accuracy无法衡量模型性能的，因此，AUC指标应运而生。

### precision和recall
在使用AUC之前，这两个指标也需要理解。

![image-20191228173720383](/Users/happyelements/Library/Application Support/typora-user-images/image-20191228173720383.png)



借用这张图说明一下，True Positive（TP）, True Negative（TN） 都是正确的预测。False Positive（FP）, False Negative（FN） 都是错误的预测。

precision：在全部被预测为正例（P）的数据中，有多少是真正的正例（TP）？即TP/(TP + FP)，主要用来预测的准不准确了，体现为"准"。
recall：全部正例（P）中，有多少被正确的预测？即TP/(TP + FN) ，主要用来观察在全部正例集合中，有多少能被预测出来，体现为"全"。

如果实在记不住英文单词，我也见过一些不错的翻译。accuracy翻译为准确率，这个比较好理解。precision和recall分别是查准率和查全率，这两个指标放在一起比较好记。至于把recall翻译成召回率，我是怎么都想不通的
。



## AUC的例子

以检测游戏中的作弊玩家为例，模型的输出是0～1的闭区间中的一个数字yy，我们认为当yy大于某个threshold值时，该玩家被预测为作弊玩家。反之，yy小于该threshold值，该玩家为非作弊玩家。

| 用户id  | 真实值y | 预测值yy   |
| ------- | ------- | ---------- |
| ***1*** | ***P*** | ***0.95*** |
| ***2*** | ***P*** | ***0.85*** |
| 3       | N       | 0.85       |
| 4       | N       | 0.80       |
| 5       | N       | 0.75       |
| 6       | N       | 0.71       |
| ***7*** | ***P*** | ***0.7***  |
| 8       | N       | 0.65       |
| 9       | N       | 0.6        |
| 10      | N       | 0.55       |
| 11      | N       | 0.5        |
| 12      | N       | 0.5        |
| 13      | N       | 0.4        |
| 14      | N       | 0.4        |
| 15      | N       | 0.4        |
| 16      | N       | 0.3        |
| 17      | N       | 0.3        |
| 18      | N       | 0.3        |
| 19      | N       | 0.2        |
| 20      | N       | 0.1        |

我们选用不同的threshold，得到的accuracy，recall，precision都是不一样的。在二分类问题中，输出yy的值介于0～1之间，但是谁也没说threshold的值必须是0.5。比如我可以假设threshold=0.7，上面表格中加黑斜体的部分就是TP，用户id3，4，5，6就是FP。套用前面介绍过的公式，当threshold=0.7时，

- accuracy =（3 + 13）/20 = 0.8
- recall = 3 / 3 = 1
- precision = 3 / （3+4）= 0.43

AUC就是为了反映在不同的threshold下，得到的TP rate（TPR） 和 FP rate（FPR）的情况。具体来说，被我简称为AUC的指标，其实应该是AUC-ROC，ROC代表（Receiver Operating Characteristics）。 如图，ROC是通过将不同threshold的x轴（FPR）和y轴（TPR）的点连接而成的曲线， AUC就是ROC下的面积。其中

- TPR = TP / P = recall
- FPR = FP / N  （某种程度上，FPR是错误分类的"recall"）

![image-20191228193855188](/Users/happyelements/Library/Application Support/typora-user-images/image-20191228193855188.png)

那么，我们设置不同的threshold，可以得到如下结果

| threshold | TPR  | FPR  |
| --------- | ---- | ---- |
| 0.95      | 0.33 | 0.   |
| 0.85      | 0.67 | 0.06 |
| 0.7       | 1    | 0.24 |
| 0.5       | 1    | 0.53 |
| 0.3       | 1    | 0.88 |
| 0.15      | 1    | 1    |
| 0.05      | 1    | 1    |

将FPR当作x轴，TPR当作Y轴，可以绘制出

![image-20191228194647666](/Users/happyelements/Library/Application Support/typora-user-images/image-20191228194647666.png)

上面的曲线，就是ROC曲线。TPR，FPR的值也都是0～1之间，故AUC的最大值为1。在实际使用中，类似tensorflow/sklearn/R之类的类库或语言，都会自动调整threshold来计算AUC。最终得到的AUC数值，即ROC曲线下的面积，AUC越接近于1，说明这个二分类器越好。

综上所述，我们可以得到一些结论：

- 上图中，threshold=0.7和0.5时，具有同样的TPR。但是threshold=0.7的FPR更低一些，所以threshold=0.7是最优的选择。这也就是说AUC-ROC可以帮我们寻找到最优的threshold。
- AUC能解决两个分类的数据不平衡的问题，虽然我没有构造极端的数据，但是不难想象，如果模型无脑预测全部结果为N或P，TPR和FPR即全为0或全为1（ROC是从坐标（0，0）到（1，1）的线段），此时绘制出的AUC面积仍然是0.5，证明二分类器毫无作用。
- AUC用于二分类器
- 没有万能的指标去评价模型，必须结合业务场景，选取最适合的指标。




参考文献：
1. [https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)
2. [https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)
3. [http://fastml.com/what-you-wanted-to-know-about-auc/](http://fastml.com/what-you-wanted-to-know-about-auc/)
4. https://medium.com/datadriveninvestor/accuracy-trap-pay-attention-to-recall-precision-f-score-auc-d02f28d3299c