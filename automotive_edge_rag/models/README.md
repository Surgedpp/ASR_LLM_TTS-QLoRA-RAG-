---
license: apache-2.0
pipeline_tag: sentence-similarity
tags:
- Sentence Transformers
- sentence-similarity
- sentence-transformers
datasets:
- shibing624/nli_zh
language:
- zh
library_name: sentence-transformers
---


# shibing624/text2vec-base-chinese
This is a CoSENT(Cosine Sentence) model: shibing624/text2vec-base-chinese.

It maps sentences to a 768 dimensional dense vector space and can be used for tasks 
like sentence embeddings, text matching or semantic search.


## Evaluation
For an automated evaluation of this model, see the *Evaluation Benchmark*: [text2vec](https://github.com/shibing624/text2vec)

- chinese text matching task：

| Arch       | BaseModel                         | Model                                                                                                                                             | ATEC  |  BQ   | LCQMC | PAWSX | STS-B | SOHU-dd | SOHU-dc |    Avg    |  QPS  |
|:-----------|:----------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-------:|:-------:|:---------:|:-----:|
| Word2Vec   | word2vec                          | [w2v-light-tencent-chinese](https://ai.tencent.com/ailab/nlp/en/download.html)                                                                    | 20.00 | 31.49 | 59.46 | 2.57  | 55.78 |  55.04  |  20.70  |   35.03   | 23769 |
| SBERT      | xlm-roberta-base                  | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | 18.42 | 38.52 | 63.96 | 10.14 | 78.90 |  63.01  |  52.28  |   46.46   | 3138  |
| Instructor | hfl/chinese-roberta-wwm-ext       | [moka-ai/m3e-base](https://huggingface.co/moka-ai/m3e-base)                                                                                       | 41.27 | 63.81 | 74.87 | 12.20 | 76.96 |  75.83  |  60.55  |   57.93   | 2980  |
| CoSENT     | hfl/chinese-macbert-base          | [shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)                                                       | 31.93 | 42.67 | 70.16 | 17.21 | 79.30 |  70.27  |  50.42  |   51.61   | 3008  |
| CoSENT     | hfl/chinese-lert-large            | [GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese)                                                   | 32.61 | 44.59 | 69.30 | 14.51 | 79.44 |  73.01  |  59.04  |   53.12   | 2092  |
| CoSENT     | nghuyong/ernie-3.0-base-zh        | [shibing624/text2vec-base-chinese-sentence](https://huggingface.co/shibing624/text2vec-base-chinese-sentence)                                     | 43.37 | 61.43 | 73.48 | 38.90 | 78.25 |  70.60  |  53.08  |   59.87   | 3089  |
| CoSENT     | nghuyong/ernie-3.0-base-zh        | [shibing624/text2vec-base-chinese-paraphrase](https://huggingface.co/shibing624/text2vec-base-chinese-paraphrase)                                 | 44.89 | 63.58 | 74.24 | 40.90 | 78.93 |  76.70  |  63.30  |    63.08  | 3066  |
| CoSENT     | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  | [shibing624/text2vec-base-multilingual](https://huggingface.co/shibing624/text2vec-base-multilingual)                                             | 32.39 | 50.33 | 65.64 | 32.56 | 74.45 |  68.88  |  51.17  |   53.67   | 4004  |


说明：
- 结果评测指标：spearman系数
- `shibing624/text2vec-base-chinese`模型，是用CoSENT方法训练，基于`hfl/chinese-macbert-base`在中文STS-B数据训练得到，并在中文STS-B测试集评估达到较好效果，运行[examples/training_sup_text_matching_model.py](https://github.com/shibing624/text2vec/blob/master/examples/training_sup_text_matching_model.py)代码可训练模型，模型文件已经上传HF model hub，中文通用语义匹配任务推荐使用
- `shibing624/text2vec-base-chinese-sentence`模型，是用CoSENT方法训练，基于`nghuyong/ernie-3.0-base-zh`用人工挑选后的中文STS数据集[shibing624/nli-zh-all/text2vec-base-chinese-sentence-dataset](https://huggingface.co/datasets/shibing624/nli-zh-all/tree/main/text2vec-base-chinese-sentence-dataset)训练得到，并在中文各NLI测试集评估达到较好效果，运行[examples/training_sup_text_matching_model_jsonl_data.py](https://github.com/shibing624/text2vec/blob/master/examples/training_sup_text_matching_model_jsonl_data.py)代码可训练模型，模型文件已经上传HF model hub，中文s2s(句子vs句子)语义匹配任务推荐使用
- `shibing624/text2vec-base-chinese-paraphrase`模型，是用CoSENT方法训练，基于`nghuyong/ernie-3.0-base-zh`用人工挑选后的中文STS数据集[shibing624/nli-zh-all/text2vec-base-chinese-paraphrase-dataset](https://huggingface.co/datasets/shibing624/nli-zh-all/tree/main/text2vec-base-chinese-paraphrase-dataset)，数据集相对于[shibing624/nli-zh-all/text2vec-base-chinese-sentence-dataset](https://huggingface.co/datasets/shibing624/nli-zh-all/tree/main/text2vec-base-chinese-sentence-dataset)加入了s2p(sentence to paraphrase)数据，强化了其长文本的表征能力，并在中文各NLI测试集评估达到SOTA，运行[examples/training_sup_text_matching_model_jsonl_data.py](https://github.com/shibing624/text2vec/blob/master/examples/training_sup_text_matching_model_jsonl_data.py)代码可训练模型，模型文件已经上传HF model hub，中文s2p(句子vs段落)语义匹配任务推荐使用
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`模型是用SBERT训练，是`paraphrase-MiniLM-L12-v2`模型的多语言版本，支持中文、英文等
- `w2v-light-tencent-chinese`是腾讯词向量的Word2Vec模型，CPU加载使用，适用于中文字面匹配任务和缺少数据的冷启动情况

## Usage (text2vec)
Using this model becomes easy when you have [text2vec](https://github.com/shibing624/text2vec) installed:

```
pip install -U text2vec
```

Then you can use the model like this:

```python
from text2vec import SentenceModel
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']

model = SentenceModel('shibing624/text2vec-base-chinese')
embeddings = model.encode(sentences)
print(embeddings)
```

## Usage (HuggingFace Transformers)
Without [text2vec](https://github.com/shibing624/text2vec), you can use the model like this: 

First, you pass your input through the transformer model, then you have to apply the right pooling-operation on-top of the contextualized word embeddings.

Install transformers:
```
pip install transformers
```

Then load model and predict:
```python
from transformers import BertTokenizer, BertModel
import torch

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = BertTokenizer.from_pretrained('shibing624/text2vec-base-chinese')
model = BertModel.from_pretrained('shibing624/text2vec-base-chinese')
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
print("Sentence embeddings:")
print(sentence_embeddings)
```

## Usage (sentence-transformers)
[sentence-transformers](https://github.com/UKPLab/sentence-transformers) is a popular library to compute dense vector representations for sentences.

Install sentence-transformers:
```
pip install -U sentence-transformers
```

Then load model and predict:

```python
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("shibing624/text2vec-base-chinese")
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']

sentence_embeddings = m.encode(sentences)
print("Sentence embeddings:")
print(sentence_embeddings)
```

## Model speed up


| Model                                                                                                                        | ATEC              | BQ                | LCQMC            | PAWSX            | STSB             |
|------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|------------------|------------------|------------------|
| shibing624/text2vec-base-chinese (fp32, baseline)                                                                            | 0.31928           | 0.42672           | 0.70157          | 0.17214          | 0.79296          |
| shibing624/text2vec-base-chinese (onnx-O4, [#29](https://huggingface.co/shibing624/text2vec-base-chinese/discussions/29))    | 0.31928           | 0.42672           | 0.70157          | 0.17214          | 0.79296          |
| shibing624/text2vec-base-chinese (ov, [#27](https://huggingface.co/shibing624/text2vec-base-chinese/discussions/27))         | 0.31928           | 0.42672           | 0.70157          | 0.17214          | 0.79296          |
| shibing624/text2vec-base-chinese (ov-qint8, [#30](https://huggingface.co/shibing624/text2vec-base-chinese/discussions/30))   | 0.30778 (-3.60%)  | 0.43474 (+1.88%)  | 0.69620 (-0.77%) | 0.16662 (-3.20%) | 0.79396 (+0.13%) |

In short:
1. ✅ shibing624/text2vec-base-chinese (onnx-O4), ONNX Optimized to [O4](https://huggingface.co/docs/optimum/en/onnxruntime/usage_guides/optimization) does not reduce performance, but gives a [~2x speedup](https://sbert.net/docs/sentence_transformer/usage/efficiency.html#benchmarks) on GPU.
2. ✅ shibing624/text2vec-base-chinese (ov), OpenVINO does not reduce performance, but gives a 1.12x speedup on CPU.
3. 🟡 shibing624/text2vec-base-chinese (ov-qint8), int8 quantization with OV incurs a small performance hit on some tasks, and a tiny performance gain on others, when quantizing with [Chinese STSB](https://huggingface.co/datasets/PhilipMay/stsb_multi_mt). Additionally, it results in a [4.78x speedup](https://sbert.net/docs/sentence_transformer/usage/efficiency.html#benchmarks) on CPU.

- usage: shibing624/text2vec-base-chinese (onnx-O4), for gpu
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "shibing624/text2vec-base-chinese",
    backend="onnx",
    model_kwargs={"file_name": "model_O4.onnx"},
)
embeddings = model.encode(["如何更换花呗绑定银行卡", "花呗更改绑定银行卡", "你是谁"])
print(embeddings.shape)
similarities = model.similarity(embeddings, embeddings)
print(similarities)
```


- usage: shibing624/text2vec-base-chinese (ov), for cpu
```python
# pip install 'optimum[openvino]'

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "shibing624/text2vec-base-chinese",
    backend="openvino",
)

embeddings = model.encode(["如何更换花呗绑定银行卡", "花呗更改绑定银行卡", "你是谁"])
print(embeddings.shape)
similarities = model.similarity(embeddings, embeddings)
print(similarities)
```

- usage: shibing624/text2vec-base-chinese (ov-qint8), for cpu
```python
# pip install optimum
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "shibing624/text2vec-base-chinese",
    backend="onnx",
    model_kwargs={"file_name": "model_qint8_avx512_vnni.onnx"},
)
embeddings = model.encode(["如何更换花呗绑定银行卡", "花呗更改绑定银行卡", "你是谁"])
print(embeddings.shape)
similarities = model.similarity(embeddings, embeddings)
print(similarities)
```


## Full Model Architecture
```
CoSENT(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_mean_tokens': True})
)
```

## Intended uses

Our model is intented to be used as a sentence and short paragraph encoder. Given an input text, it ouptuts a vector which captures 
the semantic information. The sentence vector may be used for information retrieval, clustering or sentence similarity tasks.

By default, input text longer than 256 word pieces is truncated.


## Training procedure

### Pre-training 

We use the pretrained [`hfl/chinese-macbert-base`](https://huggingface.co/hfl/chinese-macbert-base) model. 
Please refer to the model card for more detailed information about the pre-training procedure.

### Fine-tuning 

We fine-tune the model using a contrastive objective. Formally, we compute the cosine similarity from each 
possible sentence pairs from the batch.
We then apply the rank loss by comparing with true pairs and false pairs.

#### Hyper parameters

- training dataset: https://huggingface.co/datasets/shibing624/nli_zh
- max_seq_length: 128
- best epoch: 5
- sentence embedding dim: 768



## Citing & Authors
This model was trained by [text2vec](https://github.com/shibing624/text2vec). 
        
If you find this model helpful, feel free to cite:
```bibtex 
@software{text2vec,
  author = {Xu Ming},
  title = {text2vec: A Tool for Text to Vector},
  year = {2022},
  url = {https://github.com/shibing624/text2vec},
}
```