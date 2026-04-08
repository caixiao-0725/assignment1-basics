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

这里bpe tokenizer 不直接查表呢，还要一个一个合并?

BPE 的词表不是“最长匹配字典树”那种静态分词规则，而是由一条有顺序的 merges 规则定义的。例如 h e l l o，你既可以先合 h+e，也可以先合 l+l。不同先后会影响后续能不能变成 hello。 merge是有优先级的。
