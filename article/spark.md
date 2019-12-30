---
layout:     post
title:      "理解机器学习中的偏差与方差"
subtitle:   ""
date:       2018-11-17
author:     "Cactus"
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - 大数据
    - spark
---

这段时间，R同学速成spark之后，扔过来一段spark app，每天定时的运行（崩溃）在EMR集群中。这个spark app主要负责离线处理数据，刚部署在集群中时，消耗500 core， 4T内存，每天4个小时可以完成任务。没过多久，随着数据变化，执行时间增常到12个小时以上，spark程序崩溃更是家常便饭，经常需要无数次retry才能成功。本文用来记录调试，优化该spark app的点点滴滴。



#### 搞懂程序执行逻辑

刚拿到一个spark程序，肯定是一脸懵比的。在动手优化前，请务必搞清程序真正想要实现的逻辑。以R同学的程序为例，洋洋洒洒上千行，其实只是做了几件简单的事。

- 读取HDFS上的数据，大约12T，称为user behaviour
- 读取用户特征数据，大约30MB，称为features
- 根据某些特定字段，将user behaviour表和features表 JOIN在一起，称为df
- 使用一大堆map，filter过滤，筛选df
- 输出1：80%的df作为训练集
- 输出2: 20%的df作为验证集
- 输出3: 输出df的describe信息

是不是很简单的程序？写起来简单，但是实际当中，各种不靠谱的大佬会给你发过来各种建议，在迫于他们的压力开始不靠谱的优化前，请务必理清思路，不要盲目开始优化。

当然，有一些工具和方法，可以帮我们更好的理解spark代码

- 使用IntelliJ + scala插件，在本地调试，观察数据结构
- 使用dataframe.explain()方法，观察spark任务的执行过程
- 借助spark web ui观察spark任务的执行过程



#### 资源瓶颈

大佬们深知spark的参数不好调，程序时而crash，时而正常运行，那一定是参数没调对喽～

实际上，优化的第一步，先问自己几个问题

- 输入的数据有多大？这决定了硬盘和内存的使用量
- 你希望程序多久可以执行完？可别说越快越好～

看R同学的程序，输入12T数据，解压后120T。我们拥有60台Node，每台Node 250G硬盘，总共15T硬盘，看起来硬盘是够的。那我们就从spark 的log 开始分析，观察一下最先开始出现错误的地方

```
yarn application -applicationId xxxxx -containerId xxxxx | grep -C 20 ERROR | head -100
```

我发现了此类错误

```
java.io.IOException: Failed to connect to /10.249.0.96:36921
Caused by: io.netty.channel.AbstractChannel$AnnotatedConnectException: Connection refused: /10.249.0.96:36921

```

程序运行好好的，为什么会突然 connect不上呢？第一反应当然是去看看监控图表，这台机器在那个时间有什么异常没。图表显示，内存，CPU都正常，但是硬盘使用了220G以上。我们有250G硬盘，似乎还有30G没有用啊？

如果此时，你用`yarn top`命令查看，就会发现原有的60台Node，只剩下59台是healthy状态了，1台变成了unhealthy状态。那台unhealthy的Node，恰好就是上面那个IP。因为yarn认为，硬盘使用量超过90%的，就是unhealthy节点。而输入我们spark程序的数据，每个文件大小并不一致，大的几百MB，小的只有几MB，这就是数据倾斜啦。如果很不巧，几个大文件被调度到同一台节点，导致该节点变成unhealthy状态，这些大文件再被调度到另外一个倒霉的节点，反复下去，造成雪崩，整个集群的节点全部unhealthy，导致任务失败。运气好的话，大文件，小文件被适当的调度到节点，这就是程序有时会运行成功的原因。

所以，在硬盘的使用上，不要太抠了，硬盘不贵，扩容硬盘解决这个偶发crash的问题。

此外，yarn在application的级别上设置了重试次数，默认不是1。在程序未被优化前，最好设置

```
yarn.resourcemanager.am.max-attempts = 1
```

即不让yarn进行application级别的重试。因为一个application失败了，往往说明这个application有问题，重试是毫无意义的。



#### 调整参数

看了spark的文档，参数那么一大堆，刚一开始都要调整哪些参数呢？其实重点的没几个，在开始调参之前，务必搞清几个概念。

- 什么是executor
- 什么是task
- 什么是partition

这些概念建议看官方文档。简单的说，一个spark app提交后，spark app会创建N个executor，executor并不真正的执行任务，每个executor再创建N个task去真正执行任务，每个task一次只能处理一个partition的数据。由此观之——

- spark app = 老板
- executor = 小队长
- task = 干活的（你）

从数据的角度看，如果partition太多，会导致每个partition的size很小，每个task很快执行完，调度task的时间比重就过大。反正partition的size过大，会导致OOM，task没有充分并行等问题。所以，partition的数量是很重要的，可以设置`spark.sql.shuffle.partitions`并配合repartition方法，来改变partition的数量。

下面开始调整CPU和内存，假设我们有60台8core 64G的Nodes

- 为每台Node预留1core，用于管理task和Node的日常开销。所以
  spark.executor.cores = 8 -1 = 7
- 总共60个Node，预留1台用做application master，故
  spark.executor.instances = 60 -1 = 59
- 虽然Node号称64G，正确的说法是64Gi。云服务商往往用1000为单位计算，而我们所说的内存容量往往是1024为单位，实际可用内存在57G左右。借用[这篇文章](https://aws.amazon.com/cn/blogs/big-data/best-practices-for-successfully-managing-memory-for-apache-spark-applications-on-amazon-emr/)的图，我们还要为MemoryOverhead预留10%的内存。所以
  spark.executor.memory = 50G
  ![image-20190925173420947](/Users/happyelements/Library/Application Support/typora-user-images/image-20190925173420947.png)
- 最后别忘了设置 spark.driver.cores 和 spark.driver.memory，保持和exector一致就好了。

这里的设置方法，都是和参考文献1学的，强烈阅读一下原文。



至此，重要的参数已经设置完成。当然还有序列化，GC等参数，在做一些对性能极端敏感的程序时，参数调优，肯定对程序性能是有影响的。但是别忘了R同学目前面对的情况，一个要执行12个小时的spark app，executor到底是用3core还是5core，这些并不重要了。只要充分利用了资源，才方便我们去寻找真正的瓶颈。



#### 性能瓶颈

经过"调优"的程序，跑了没几天，源数据略有增长，又crash了。现在要命的是每次必crash，而不是偶发crash。R同学很着急，大佬很生气。R同学要买更多的Node，拜托，现在一个月的服务器成本已经够我半年的工资了，我看着都心疼。

- 先查看crash的log吧

```
ExecutorLostFailure (executor 57 exited caused by one of the running tasks) Reason: Container killed by YARN for exceeding memory limits. 53.6 GB of 53 GB physical memory used. Consider boosting spark.yarn.executor.memoryOverhead.
```

按照错误信息所言，提高了 spark.yarn.executor.memoryOverhead 的数量，仍然crash。这个值到底要设置多少才够呢？这么想是无解的，还是应该从数据出发，我们看看数据到底有多大。

![image-20190925175531882](/Users/happyelements/Library/Application Support/typora-user-images/image-20190925175531882.png)

通过spark ui，我们发现报错的这一步，出现了严重的数据倾斜。虽然买足够多的内存也能解决问题，但还是帮老板省省吧。回顾前文所述，这个数据倾斜发生在JOIN大数据表user behaviour 和feature。feature只有30多MB，我们完全可以把feature表broadcast到每一个executor中，省去了各种shuffle read/write的时间。有关join优化的话题，网上有很多，推荐参考文献里美团技术团队的文章。

这里还有个小插曲，R同学本打算在这里使用broadcast的JOIN（R同学并不是那么菜），所以他设置了

```
spark.sql.conf.autoBroadcastJoinThreshold	= 100000000
```

大意是100MB以下的表，在JOIN时，会自动broadcast。

但是，在官网上，还有这么一句

```
The BROADCAST hint guides Spark to broadcast each specified table when joining them with another table or view. When Spark deciding the join methods, the broadcast hash join (i.e., BHJ) is preferred, even if the statistics is above the configuration spark.sql.autoBroadcastJoinThreshold. When both sides of a join are specified, Spark broadcasts the one having the lower statistics. Note Spark does not guarantee BHJ is always chosen, since not all cases (e.g. full outer join) support BHJ. When the broadcast nested loop join is selected, we still respect the hint
```

大意是即便你设置了autoBroadcastJoinThreshold，也未必会auto broadcast。。。所以，请务必在代码中显示的写出broadcast方法。



- 到这里，解决掉另一个crash问题。但是程序执行仍需要12个小时，这合理么？

有关程序耗时是否合理，其实可以大概估算一下。60台Node，总共420个task并行处理12T的压缩数据，耗时12个小时。总耗时420 * 12 = 5040 小时 处理 12T压缩数据，平均处理1G数据需要花费25分钟。以当前的业务来看，无疑是耗时过长了，这里面很可能存在问题。

于是，我通过spark ui重点观察一下那几个最耗时的stage的情况。什么？这几个stage的输入都是shuffle read 12T？！Oh my god！R同学犯了一个严重的错误，由于3次输出，都是基于相同的transform的。spark的RDD默认情况下是不会持久化或缓存任何中间数据的，需要你手动搞定，调用persist方法即可。

![image-20190925181553975](/Users/happyelements/Library/Application Support/typora-user-images/image-20190925181553975.png)

伪代码如下：

```
    val dfTransformed = df
      .transform(transform_xxx).persist(StorageLevel.DISK_ONLY)

    dfTransformed.output1()
    dfTransformed.output2()
    dfTransformed.output3()
```

如果不调用persist，transform是会被从头执行多次的，这和spark的transform和action的机制有关。



至此，该程序的执行时间已经缩短70%以上，并且平稳运行。优化还远没有结束，希望明天不要再crash～




参考文档：

- https://aws.amazon.com/cn/blogs/big-data/best-practices-for-successfully-managing-memory-for-apache-spark-applications-on-amazon-emr/
- https://tech.meituan.com/2016/04/29/spark-tuning-basic.html
- https://tech.meituan.com/2016/05/12/spark-tuning-pro.html