# CS336 Spring 2025 Assignment 1: Basics


For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```


## 作业总结

纯在Ai帮助下写的，把约束MD文件直接删了！但是下面的内容纯手打

bpe training是给你一个很大的文本，比如安徒生童话，你从里面提取出merge对，就是很常用的词缀。然后 bpe tokenizer是借助刚才得到的词汇表还有merge对，把文本转成数字，这样就可以喂给神经网络。

## 问ai的一些傻子问题

### bpe tokenizer 不直接查表呢，还要一个一个合并?

BPE 的词表不是“最长匹配字典树”那种静态分词规则，而是由一条有顺序的 merges 规则定义的。例如 h e l l o，你既可以先合 h+e，也可以先合 l+l。不同先后会影响后续能不能变成 hello。 merge是有优先级的。

### nn.Parameter的作用是什么？

requires_grad=True 默认打开。反向传播会给这个张量算梯度，优化器可以更新它（普通 torch.tensor 放进 Module 里默认不会自动这样）。

会登记到 nn.Module 上。赋值给 self.weight 时，模块会把 weight 收进 parameters() / named_parameters()，这样 model.parameters()、state_dict()、load_state_dict()、to(device) 等都会自动带上它。

和 buffer 不同。register_buffer 的是非训练状态（如 running mean）；Parameter 表示可训练权重。

### 我一直觉得在transformer中q k v的计算是最昂贵的，然后ffn更像个辅助的，让模型不爆炸的工具，ffn这么吃资源它配吗？

FFN 不仅 “配”，而且它才是 Transformer 真正的 “大脑”。
Attention 只是 “查找信息”，FFN 才是 “思考与记忆”。

Attention = 查资料、抄笔记、找上下文。

FFN = 真正的思考、推理、记忆、理解、归纳。

<完>