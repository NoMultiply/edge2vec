[Edge2vec: Edge-based Social Network Embedding](http://ise.thss.tsinghua.edu.cn/~wangchaokun/edge2vec/tkdd_embedding_accepted.pdf)
====================================

**Edge2vec** is the first edge-based graph embedding method to map the edges in social networks directly to low-dimensional vectors. It is designed to preserve both the local and the global structure information of edges and the learned representation vectors can be applied to various tasks such as link prediction, social tie direction prediction and social tie sign prediction.

[See more about Edge2vec.](http://ise.thss.tsinghua.edu.cn/~wangchaokun/edge2vec/edge2vec.html)

--------------------------------------------------

The code is written in python3 using the `tensorflow` framework. Other libs can be found in the file `requirements.txt`.

#### Usage
To run edge2vec, open the terminal and input
```bash
python3 edge2vec.py -i INPUT -m MODEL -n NUM -s SAMPLE
```
, where the parameters is explained as follows:
```
-i: path to the input graph file (in "edge list" format)
-m: the output directory of model files
-n: the maximum num of the node
-s: the num of negative samples
```

For example, you can run edge2vec on `Epinions` using

```bash
python edge2vec.py -i Epinions-55K.graph -m results -n 1000 -s 500
```

The program will divide the input graph into two part, `MODEL/train.txt` and `MODEL/test.txt`, and their embedding results can be found in `MODEL/train.log` and `MODEL/test.log`.  You can use these to conduct downstream experiment, such as link prediction, sign prediction or tie direction prediction.

#### References
Chanping Wang, Chaokun Wang, Zheng Wang, Xiaojun Ye, and PhilipS. Yu. Edge2vec: Edge-based Social Network Embedding. ACM Transactions on Knowledge Discovery in Data (TKDD), 14(4):1-24, 2020. (Submitted June 2017; accepted March 2020)
