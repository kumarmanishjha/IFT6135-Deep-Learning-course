import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).
    
    inputs: 
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1
class RNNCell(nn.Module):
    """
    Our implementation of an RNN cell with dropout and tanh nonlinearity
    """
    def __init__(self, input_size, hidden_size, dp_keep_prob):
        """
        input_size:   The number of expected features in the input
        hidden_size:  The number of features in the hidden state
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
        """
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wx = nn.Linear(input_size, hidden_size, bias=False)  # W_x * x_t
        self.Wh = nn.Linear(hidden_size, hidden_size)              # W_h * h_t + b_h
        self.dropout = nn.Dropout(1 - dp_keep_prob)
        self.tanh = nn.Tanh()

    def init_weights_uniform(self):
        """
        initializes the weights
        """
        k = np.sqrt(1/self.hidden_size)
        nn.init.uniform_(self.Wx.weight.data, a=-k, b=k)
        nn.init.uniform_(self.Wh.weight.data, a=-k, b=k)
        nn.init.uniform_(self.Wh.bias.data, a=-k, b=k)

    def forward(self, x, h):
        """
        Arguments:
          - x: batch_size * emb_size
          - h: batch_size * hidden_size
        Returns:
          - h: batch_size * hidden_size
        """
        h = self.dropout(h)  # apply dropout
        h = self.tanh(self.Wx(x) + self.Wh(h))  # h_(t+1) = tanh(W_x*x_(t+1) + W_h*h_t + b_h)
        #y = self.dropout(h)  # apply dropout

        #return y, h
        return h

class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, 
                 vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the 
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.

        """
        super(RNN, self).__init__()
        # parameters
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob
        # embedding
        self.emb_layer = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.emb_size)
        #self.emb_dropout = nn.Dropout(1 - self.dp_keep_prob)
        # list of hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(self.num_layers):
            input_size = self.emb_size if i == 0 else self.hidden_size
            hidden_layer = RNNCell(input_size, self.hidden_size, self.dp_keep_prob)
            self.hidden_layers.append(hidden_layer)
        # output layer
        self.output_dropout = nn.Dropout(1 - self.dp_keep_prob)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size, bias=True)
        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
        and output biases to 0 (in place). The embeddings should not use a bias vector.
        Initialize all other (i.e. recurrent and linear) weights AND biases uniformly 
        in the range [-k, k] where k is the square root of 1/hidden_size
        """
        # embedding
        nn.init.uniform_(self.emb_layer.weight.data, a=-0.1, b=0.1)
        # hidden
        for hidden_layer in self.hidden_layers:
            hidden_layer.init_weights_uniform()
        # output
        nn.init.uniform_(self.output_layer.weight.data, a=-0.1, b=0.1)
        nn.init.zeros_(self.output_layer.bias.data)

    def init_hidden(self):
        """
        Initialize the hidden states to zero
        This is used for the first mini-batch in an epoch, only.
        """
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        """
        Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that 
                    represent the index of the current token(s) in the vocabulary.
                        shape: (seq_len, batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
        - Logits for the softmax over output tokens at every time-step.
              **Do NOT apply softmax to the outputs!**
              Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
              this computation implicitly.
                    shape: (seq_len, batch_size, vocab_size)
        - The final hidden states for every layer of the stacked RNN.
              These will be used as the initial hidden states for all the 
              mini-batches in an epoch, except for the first, where the return 
              value of self.init_hidden will be used.
              See the repackage_hiddens function in ptb-lm.py for more details, 
              if you are curious.
                    shape: (num_layers, batch_size, hidden_size)
        """
        # outputs at each time step
        logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size],
                             device=inputs.device)
        # input embedding
        emb_inputs = self.emb_layer(inputs)
        #emb_inputs = self.emb_dropout(emb_inputs)
        # loop over time steps
        for t in range(self.seq_len):
            # input at this time step
            layer_input = emb_inputs[t]
            hidden_next_list = []  # next timestep
            # loop over layers
            for i, hidden_layer in enumerate(self.hidden_layers):
                hidden_out = hidden_layer(layer_input, hidden[i])  # output of hidden layer
                layer_input = hidden_out  # feed as the input for the next layer
                hidden_next_list.append(hidden_out)  # add to the list
            # stack to get the hidden layers for this timestep
            hidden = torch.stack(hidden_next_list)
            # output at this timestep
            logits[t] = self.output_layer(self.output_dropout(layer_input))
        
        return logits, hidden  # for last timestep

    def generate(self, input, hidden, generated_seq_len):
        """
        Arguments:
        - input: A mini-batch of input tokens (NOT sequences!)
                        shape: (batch_size)
        - hidden: The initial hidden states for every layer of the stacked
                  RNN.
                        shape: (num_layers, batch_size, hidden_size)
        - generated_seq_len: The length of the sequence to generate.
                       Note that this can be different than the length used 
                       for training (self.seq_len)
        Returns:
        - Sampled sequences of tokens
                    shape: (generated_seq_len, batch_size)
        """
        tokens = input.view(1, -1)  # reshape
        emb_inputs = self.emb_layer(tokens)  # embedding
        #emb_inputs = self.emb_dropout(emb_inputs)  # dropout
        # loop over time step
        for t in range(generated_seq_len):
            # input at this time step
            layer_input = emb_inputs[0]
            hidden_next_list = []  # next timestep
            # loop over layers
            for i, hidden_layer in enumerate(self.hidden_layers):
                hidden_out = hidden_layer(layer_input, hidden[i])  # output of hidden layer
                layer_input = hidden_out  # feed as the input for the next layer
                hidden_next_list.append(hidden_out)  # add to the list
            # stack to get the hidden layers for this timestep
            hidden = torch.stack(hidden_next_list)
            # output at this timestep
            logits = self.output_layer(self.output_dropout(layer_input)).detach()
            probs = F.softmax(logits, dim=1)
            # append output
            sample = Categorical(probs=probs).sample().view(1, -1)
            tokens = torch.cat((tokens, sample), dim=0)
            # embed to get next input
            emb_inputs = self.emb_layer(tokens)  # embedding
            #emb_inputs = self.emb_dropout(emb_inputs)  # dropout

        return tokens

# Problem 2
class GRUCell(nn.Module):
    """
    Our implementation of an GRU cell with dropout
    """
    def __init__(self, input_size, hidden_size, dp_keep_prob):
        """
        input_size:   The number of expected features in the input
        hidden_size:  The number of features in the hidden state
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
        """
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dp_keep_prob = dp_keep_prob
        self.Wr = nn.Linear(self.input_size, self.hidden_size)  # W_r * x_t + b_r
        self.Wz = nn.Linear(self.input_size, self.hidden_size)  # W_z * x_t + b_z
        self.Wh = nn.Linear(self.input_size, self.hidden_size)  # W_h * x_t + b_h
        self.Ur = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # U_r * h_(t-1)
        self.Uz = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # U_z * h_(t-1)
        self.Uh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # U_h * (r_t h_(t-1))
        self.sigmoid_r = nn.Sigmoid()
        self.sigmoid_z = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(1 - self.dp_keep_prob)

    def init_weights_uniform(self):
        """
        initializes the weights
        """
        k = np.sqrt(1/self.hidden_size)
        nn.init.uniform_(self.Wr.weight.data, a=-k, b=k)
        nn.init.uniform_(self.Wr.bias.data, a=-k, b=k)
        nn.init.uniform_(self.Wz.weight.data, a=-k, b=k)
        nn.init.uniform_(self.Wz.bias.data, a=-k, b=k)
        nn.init.uniform_(self.Wh.weight.data, a=-k, b=k)
        nn.init.uniform_(self.Wh.bias.data, a=-k, b=k)
        nn.init.uniform_(self.Ur.weight.data, a=-k, b=k)
        nn.init.uniform_(self.Uz.weight.data, a=-k, b=k)
        nn.init.uniform_(self.Uh.weight.data, a=-k, b=k)

    def forward(self, x, h):
        """
        Arguments:
          - x: batch_size * emb_size
          - h: batch_size * hidden_size
        Returns:
          - h: batch_size * hidden_size
        """
        h = self.dropout(h)  # apply dropout
        # r_t = sigm(W_r * x_t + U_r * h_(t-1) + b_r)
        r = self.sigmoid_r(self.Wr(x) + self.Ur(h))
        # z_t = sigm(W_z * x_t + U_z * h_(t-1) + b_z)
        z = self.sigmoid_z(self.Wz(x) + self.Uz(h))
        # h_t_tilde = tanh(W_h * x_t + U_h * (r_t h_(t-1)) + b_h)
        h_tilde = self.tanh(self.Wh(x) + self.Uh(r * h))
        
        return (1 - z) * h + z * h_tilde

class GRU(nn.Module):
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, 
                 vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the 
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.

        """
        super(GRU, self).__init__()
        # parameters
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob
        # embedding
        self.emb_layer = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.emb_size)
        #self.emb_dropout = nn.Dropout(1 - self.dp_keep_prob)
        # list of hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(self.num_layers):
            input_size = self.emb_size if i == 0 else self.hidden_size
            hidden_layer = GRUCell(input_size, self.hidden_size, self.dp_keep_prob)
            self.hidden_layers.append(hidden_layer)
        # output layer
        self.output_dropout = nn.Dropout(1 - self.dp_keep_prob)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size, bias=True)
        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
        and output biases to 0 (in place). The embeddings should not use a bias vector.
        Initialize all other (i.e. recurrent and linear) weights AND biases uniformly 
        in the range [-k, k] where k is the square root of 1/hidden_size
        """
        # embedding
        nn.init.uniform_(self.emb_layer.weight.data, a=-0.1, b=0.1)
        # hidden
        for hidden_layer in self.hidden_layers:
            hidden_layer.init_weights_uniform()
        # output
        nn.init.uniform_(self.output_layer.weight.data, a=-0.1, b=0.1)
        nn.init.zeros_(self.output_layer.bias.data)

    def init_hidden(self):
        """
        Initialize the hidden states to zero
        This is used for the first mini-batch in an epoch, only.
        """
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        """
        Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that 
                    represent the index of the current token(s) in the vocabulary.
                        shape: (seq_len, batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
        - Logits for the softmax over output tokens at every time-step.
              **Do NOT apply softmax to the outputs!**
              Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
              this computation implicitly.
                    shape: (seq_len, batch_size, vocab_size)
        - The final hidden states for every layer of the stacked RNN.
              These will be used as the initial hidden states for all the 
              mini-batches in an epoch, except for the first, where the return 
              value of self.init_hidden will be used.
              See the repackage_hiddens function in ptb-lm.py for more details, 
              if you are curious.
                    shape: (num_layers, batch_size, hidden_size)
        """
        # outputs at each time step
        logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size],
                             device=inputs.device)
        # input embedding
        emb_inputs = self.emb_layer(inputs)
        #emb_inputs = self.emb_dropout(emb_inputs)
        # loop over time steps
        for t in range(self.seq_len):
            # input at this time step
            layer_input = emb_inputs[t]
            hidden_next_list = []  # next timestep
            # loop over layers
            for i, hidden_layer in enumerate(self.hidden_layers):
                hidden_out = hidden_layer(layer_input, hidden[i])  # output of hidden layer
                layer_input = hidden_out  # feed as the input for the next layer
                hidden_next_list.append(hidden_out)  # add to the list
            # stack to get the hidden layers for this timestep
            hidden = torch.stack(hidden_next_list)
            # output at this timestep
            logits[t] = self.output_layer(self.output_dropout(layer_input))
        
        return logits, hidden  # for last timestep

    def generate(self, input, hidden, generated_seq_len):
        """
        Arguments:
        - input: A mini-batch of input tokens (NOT sequences!)
                        shape: (batch_size)
        - hidden: The initial hidden states for every layer of the stacked
                  RNN.
                        shape: (num_layers, batch_size, hidden_size)
        - generated_seq_len: The length of the sequence to generate.
                       Note that this can be different than the length used 
                       for training (self.seq_len)
        Returns:
        - Sampled sequences of tokens
                    shape: (generated_seq_len, batch_size)
        """
        tokens = input.view(1, -1)  # reshape
        emb_inputs = self.emb_layer(tokens)  # embedding
        #emb_inputs = self.emb_dropout(emb_inputs)  # dropout
        # loop over time step
        for t in range(generated_seq_len):
            # input at this time step
            layer_input = emb_inputs[0]
            hidden_next_list = []  # next timestep
            # loop over layers
            for i, hidden_layer in enumerate(self.hidden_layers):
                hidden_out = hidden_layer(layer_input, hidden[i])  # output of hidden layer
                layer_input = hidden_out  # feed as the input for the next layer
                hidden_next_list.append(hidden_out)  # add to the list
            # stack to get the hidden layers for this timestep
            hidden = torch.stack(hidden_next_list)
            # output at this timestep
            logits = self.output_layer(self.output_dropout(layer_input)).detach()
            probs = F.softmax(logits, dim=1)
            # append output
            sample = Categorical(probs=probs).sample().view(1, -1)
            tokens = torch.cat((tokens, sample), dim=0)
            # embed to get next input
            emb_inputs = self.emb_layer(tokens)  # embedding
            #emb_inputs = self.emb_dropout(emb_inputs)  # dropout

        return tokens



#Question 3 ##################################################

#----------------------------------------------------------------------------------

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()

        assert n_units % n_heads == 0
        self.h = n_heads
        self.n_units = n_units
        self.d_k = self.n_units// self.h
        
        self.q_linear = nn.Linear(self.n_units, self.n_units)
        self.v_linear = nn.Linear(self.n_units, self.n_units)
        self.k_linear = nn.Linear(self.n_units, self.n_units)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.n_units, self.n_units)
        
        
        """
        initializes the weights
        """
        
        nn.init.uniform_(self.q_linear.weight.data,
                         a=-np.sqrt(1/self.n_units), 
                         b=np.sqrt(1/self.n_units))
        nn.init.uniform_(self.v_linear.weight.data,
                         a=-np.sqrt(1/self.n_units), 
                         b=np.sqrt(1/self.n_units))
        nn.init.uniform_(self.k_linear.weight.data,
                         a=-np.sqrt(1/self.n_units), 
                         b=np.sqrt(1/self.n_units))
        nn.init.uniform_(self.out.weight.data,
                         a=-np.sqrt(1/self.n_units), 
                         b=np.sqrt(1/self.n_units))
        
        nn.init.uniform_(self.q_linear.bias.data,
                         a=-np.sqrt(1/self.n_units),
                         b=np.sqrt(1/self.n_units))
        nn.init.uniform_(self.v_linear.bias.data,
                         a=-np.sqrt(1/self.n_units),
                         b=np.sqrt(1/self.n_units))        
        nn.init.uniform_(self.k_linear.bias.data,
                         a=-np.sqrt(1/self.n_units),
                         b=np.sqrt(1/self.n_units))        
        nn.init.uniform_(self.out.bias.data,
                         a=-np.sqrt(1/self.n_units),
                         b=np.sqrt(1/self.n_units))    
        
    def forward(self, query, key, value, mask=None):
        
        bs = query.size(0)
        
        # perform linear operation and split into h heads
        
        key = self.k_linear(key).view(bs, -1, self.h, self.d_k)
        query = self.q_linear(query).view(bs, -1, self.h, self.d_k)
        value = self.v_linear(value).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        key = key.transpose(1,2)
        query = query.transpose(1,2)
        value = value.transpose(1,2)
# calculate attention using function we will define next

        scores = torch.matmul(query, key.transpose(-2, -1)) /  math.sqrt(self.d_k)
        if mask is not None:
                mask = mask.unsqueeze(1)
                scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            scores = self.dropout(scores)
        
        output = torch.matmul(scores, value)
        
        # concatenate heads and put through final linear layer
        concat = output.transpose(1,2).contiguous()\
        .view(bs, -1, self.n_units)
        
        result = self.out(concat)
    
        return result


# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

