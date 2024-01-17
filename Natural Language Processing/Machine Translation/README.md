# Machine Translation

In this project, I use PyTorch to implement a sequence to sequence (Seq2Seq) network with attention to build a neural machine translation system (translating French --> English). 

## Data

Here is an example of our working data. The source language is French, while the target language is English

![](outputs/example_input)

### Data Preprocessing 

The first in  processing is inserting essential tokens at the beginning and end of sentences. This involves inserting <s> at the beginning and </s> at the end. These tokens serve as markers for the model and provide crucial context and structural cues during subsequent stages of processing and analysis.

We then add padding to the input sentences with our specified pad_token, according to the longest sentence in the batch, so that all sentences now have equal length.

### Build Vocabulary

The next step is to construct the vocabulary for the Seq2Seq Model. The approach involves a traditional setup of mapping words to indices. The importance of building a robust vocabulary lies in its role as the foundation for effective language modeling. Given that our model size is not extensive and mapping every single word to an index might result in sparsity issues, we adopt a prudent strategy—- only words surpassing a specified frequency threshold are mapped. This ensures that the vocabulary captures meaningful information while mitigating sparsity concerns, ultimately enhancing the model's efficiency and performance.


## Seq2Seq Model Implementation

In efforts to learn about the model more throughouly, I implement this Seq2Seq using various PyTorch Layers, instead of using an out of the box model.

### Model Embeddings 

First, I define a pytorch nn.module ModelEmbeddings to convert inputs words to their embedding vectors. It initializes embedding layers for source and target languages using a specified size and vocabulary. Padding is also incorporated into the embeddings.  


### Seq2Seq Model
This model is designed to complete a simple Neural Machine Translation task. I utilize a Bidirectional LSTM Encoder, a Unidirectional LSTM Decoder, and a Global Attention Model (Luong, et al. 2015).

The model takes in 4 parameters, the embedding size, the size of the hidden layer, the vocabularly containing source and target languarages, and the droprate (used for attention)

Here is my model:

```python
# Bidirectional LSTM with bias
self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=True)
# LSTM Cell with bias
self.decoder = nn.LSTMCell(embed_size + hidden_size, hidden_size)

# Linear Layer with no bias, called W_{h} in the PDF.
self.h_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
# Linear Layer with no bias, called W_{c} in the PDF.
self.c_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
# Linear Layer with no bias, called W_{attProj} in the PDF.
self.att_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
# Linear Layer with no bias, called W_{u} in the PDF.
self.combined_output_projection = nn.Linear(hidden_size * 2 + hidden_size, hidden_size, bias=False)
# Linear Layer with no bias, called W_{vocab} in the PDF.
self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False)
# Dropout Layer
self.dropout = nn.Dropout(self.dropout_rate)
```

A basic walk through of my forward pass is as follows:
1. Apply the encoder to the padded source (`source_padded`)by calling `self.encode()`
2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
3. Apply the decoder to compute combined-output by calling `self.decode()`
4. Compute log probability distribution over the target vocabulary using the combined_outputs returned by the `self.decode()` function.

**Encoder** 

The encoder is applied to source sentences to obtain encoder hidden states. Additonally, during this step I also take the final states of the encoder and project them to obtain initial states for decoder.

Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.

Compute `dec_init_state`, a tuple containing `init_decoder_hidden` and `init_decoder_cell`, concatenate the forward and backward tensors of `last_hidden` and `last_cell`. Apply the projection layers `h_projection` and `c_projection` to obtain the initial decoder hidden state and cell state, respectively.

**Decoder**

Here are the basic steps I used for decoding
1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
3. Use the torch.split function to iterate over the time dimension of Y. I call the step functionhere to compite the Decoder's next cell and state values. I update `o_prev` (previous output vector) and `o_t` accordingly.     
4. Use torch.stack to convert combined_outputs from a list length tgt_len of tensors shape (b, h), to a single tensor shape (tgt_len, b, h) where tgt_len = maximum target sentence length, b = batch size, h = hidden size.

**Forward Step**

Here is how I compute one forward step of the LSTM decoder, including the attention computation.
1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
2. Split dec_state into its two parts (dec_hidden, dec_cell)
3. Compute the attention scores e_t, a Tensor shape (b, src_len).
4.  Apply softmax to e_t to yield alpha_t
5. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the attention output vector, a_t.
6. Concatenate dec_hidden with a_t to compute tensor U_t
7. Apply the combined output projection layer to U_t to compute tensor V_t
8. Compute tensor O_t by first applying the Tanh function and then the dropout layer.


## Training Model

To train the model, I used these paramaters

BATCH_SIZE = 32
EMBED_SIZE = 64
HIDDEN_SIZE = 64
DROPOUT_RATE = 0.3

LOG_EVERY = 10
VALID_NITER = 100
PATIENCE = 5
LR = 0.001
LR_DECAY = 0.5
MAX_EPOCH = 50
MAX_NUM_TRIAL = 5


I also used the adam optimizer and in efforts to address the exploding gradient problem, I decided to use gradient clipping. This involves setting a threshold value, and if the norm (magnitude) of the gradients exceeds this threshold, the gradients are scaled down proportionally to ensure they do not surpass the threshold. inadditon to this, I also used used a decaying learning rate in efforts to improve the optimization process and achieve better convergence.

```python 
lr = optimizer.param_groups[0]['lr'] * LR_DECAY
```

When building the vocabulary for training and testing, I used a corpus size of 5000 and frequeney threshold of 2.


### Metrics
Instead of using accuracy to evaluate the model, I used a corpus level BLUE score 'from nltk.translate.bleu_score import corpus_bleu'. 

$BLEU = BP \times \exp\left(\sum_{n=1}^{4} \frac{1}{n} \cdot \log(Pn)\right)$

where:
- $BLEU$: BLEU score
- $BP$: Brevity Penalty
- $Pn$: Precision for n-grams (1 to 4, typically)
- $\exp()$: Exponential function
- $\sum()$: Summation
- $\log()$: Natural logarithm
- $n$: N-gram order

The corpus-level BLEU score aggregates the individual sentence-level BLEU scores to provide an overall evaluation metric for the entire dataset. It offers a comprehensive assessment of the translation quality across multiple sentences.


## Translations

Here are sample translations my model performed from french to english.


Input: ne suis pas une s@@ ain@@ te
Gold: i m no saint .
Pred: i m saint .

Input: etes un mer@@ ve@@ ill@@ eux ami
Gold: you re a wonderful friend .
Pred: you re a beau@@ ty

Input: n etes qu un l@@ ache
Gold: you re no@@ thing but a co@@ ward .
Pred: you you you re not good only girl can .


As you can see they aren't the best translations in all cases. ()