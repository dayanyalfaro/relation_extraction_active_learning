import os
import torch
import torch.nn as nn

from utils import load_pkl

class Embedding(nn.Module):
    def __init__(self, config):
        """
        word embedding: Generally 0 to padding
        pos embedding:  Generally 0 to padding
        dim_strategy: [cat, sum]  Whether multiple embeddings are spliced or added
        """
        super(Embedding, self).__init__()
        if config.corpus.use_pretrained:
            emb_path = os.path.join(config.cwd, config.corpus.out_path, 'word2vec.pkl')
            embedding_vectors = load_pkl(emb_path)
            self.word_dim = embedding_vectors.shape[1] # necessary for pos_dim
        else:
            self.vocab_size = config.vocab_size
            self.word_dim = config.word_dim

        self.pos_size = config.pos_size
        self.pos_dim = config.pos_dim if config.dim_strategy == 'cat' else self.word_dim
        self.dim_strategy = config.dim_strategy

        if config.corpus.use_pretrained:
            self.wordEmbed = nn.Embedding.from_pretrained(embeddings=embedding_vectors, freeze=False)
        else:
            self.wordEmbed = nn.Embedding(self.vocab_size, self.word_dim, padding_idx=0)

        self.headPosEmbed = nn.Embedding(self.pos_size, self.pos_dim, padding_idx=0)
        self.tailPosEmbed = nn.Embedding(self.pos_size, self.pos_dim, padding_idx=0)

    def forward(self, *x):
        word, head, tail = x
        word_embedding = self.wordEmbed(word)
        head_embedding = self.headPosEmbed(head)
        tail_embedding = self.tailPosEmbed(tail)

        if self.dim_strategy == 'cat':
            return torch.cat((word_embedding, head_embedding, tail_embedding), -1)
        elif self.dim_strategy == 'sum':
            # pos_dim == word_dim
            return word_embedding + head_embedding + tail_embedding
        else:
            raise Exception('dim_strategy must choose from [sum, cat]')
