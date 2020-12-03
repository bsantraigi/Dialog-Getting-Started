# Getting Started: Dialog Systems

Resources for Getting Started in NLP for Dialog Systems

## The Basic I/O structure of Dialogs: 

**Input (context):** The sequence of all previous utterances.

**Output (response):** The final utterance to be predicted.

`__eou__`: Special token for end of utterance.

```
Complete Dialog:
 can you do push-ups ? 
 of course i can . it's a piece of cake ! believe it or not , i can do 30 push-ups a minute . 
 really ? i think that's impossible ! 
 you mean 30 push-ups ? 
 yeah ! 
 it's easy . if you do exercise everyday , you can make it , too . 

Training Samples:
[1]
Input: can you do push-ups ? 
Output: of course i can . it's a piece of cake ! believe it or not , i can do 30 push-ups a minute . 

[2]
Input: can you do push-ups ? __eou__ of course i can . it's a piece of cake ! believe it or not , i can do 30 push-ups a minute . 
Output: really ? i think that's impossible ! 

[3]
Input: can you do push-ups ? __eou__ of course i can . it's a piece of cake ! believe it or not , i can do 30 push-ups a minute . __eou__ really ? i think that's impossible ! 
Output: you mean 30 push-ups ? 

[4]
Input: can you do push-ups ? __eou__ of course i can . it's a piece of cake ! believe it or not , i can do 30 push-ups a minute . __eou__ really ? i think that's impossible ! __eou__ you mean 30 push-ups ? 
Output: yeah ! 

[5]
Input: can you do push-ups ? __eou__ of course i can . it's a piece of cake ! believe it or not , i can do 30 push-ups a minute . __eou__ really ? i think that's impossible ! __eou__ you mean 30 push-ups ? __eou__ yeah ! 
Output: it's easy . if you do exercise everyday , you can make it , too .
```

# Basic Materials

**NLP Lectures:**

1. [CS224N: Natural Language Processing with Deep Learning | Winter 2019](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z) (Dan Jurafsky)
  1. L1,2,3,4
  2. L6 (Language Modeling)
  3. L8,13,14
2. [CMU Neural Nets for NLP 2019](https://www.youtube.com/playlist?list=PL8PYTP1V4I8Ajj7sY6sdtmjgkt7eo2VMs) (Graham Neubig) L5,6,7,8,18

**Pytorch + Deep Learning Tutorials:**

1. [Dive into Deep Learning — Dive into Deep Learning 0.8.0 documentation](https://d2l.ai/)
2. [fast.ai Code-First Intro to Natural Language Processing](https://www.youtube.com/playlist?list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9)
3. How to calculate **Perplexity** in Seq2Seq models when using Pytorch? Follow the method from [Sequence-to-Sequence Modeling with nn.Transformer and TorchText](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

# Great Seq2Seq Tutorials

1. [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
 It talks about the paper [[1706.03762] Attention Is All You Need](https://arxiv.org/abs/1706.03762). This presents a **Sequence to Sequence** architecture for **Neural Machine Translation**
2. **Chatbot Tutorial:** [**Chatbot Tutorial**](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
1. [Sequence-to-Sequence Modeling with nn.Transformer and TorchText](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
2. [Transformer — PyTorch master documentation](https://pytorch.org/docs/master/generated/torch.nn.Transformer.html)
3. [huggingface/transformers: 🤗Transformers: State-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0.](https://github.com/huggingface/transformers)

## Datasets:

**Daily Dialog dataset**. (Link: [Paper](https://arxiv.org/pdf/1710.03957.pdf))

**DailyDialog Preprocessing:** [**Sanghoon94/DailyDialogue-Parser: Parser for DailyDialogue Dataset**](https://github.com/Sanghoon94/DailyDialogue-Parser)

**Multiwoz:** [**https://github.com/budzianowski/multiwoz**](https://github.com/budzianowski/multiwoz)

## Other Helpful Blog Posts:

- [How do Transformers Work in NLP? A Guide to the Latest State-of-the-Art Models](https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/)
- [The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time.](http://jalammar.github.io/illustrated-transformer/)
- [How Transformers Work](https://towardsdatascience.com/transformers-141e32e69591)
- [Transformers from scratch](http://www.peterbloem.nl/blog/transformers)
- [What is a Transformer? - Inside Machine learning](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04)
- FastAI tutorial on Transformers (with code): [The Transformer for language translation (NLP video 18)](https://www.youtube.com/watch?v=KzfyftiH7R8)
- [https://github.com/fawazsammani/chatbot-transformer/blob/master](https://github.com/fawazsammani/chatbot-transformer/blob/master/models.py)PYTORCH CHATBOT TRANSFORMER IMPLEMENTATION
- Good read for various decoding techniques: [https://huggingface.co/blog/how-to-generate](https://huggingface.co/blog/how-to-generate)

