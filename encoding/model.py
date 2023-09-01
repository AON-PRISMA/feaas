import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models

from sent_trans import ContrastiveLearningST


def get_model(name, encoding_size=256):
    if name == 'bert-base-uncased':
        word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=encoding_size,
                                   activation_function=nn.Tanh())

        model = ContrastiveLearningST(modules=[word_embedding_model, pooling_model, dense_model, Normalize(p=2)])
        return model
    elif name == 'all-distilroberta-v1':
        model = SentenceTransformer('all-distilroberta-v1')
    elif name == 'all-mpnet-base-v2':
        model = SentenceTransformer('all-mpnet-base-v2')  # max_seq_len: 384
    elif name == 'multi-qa-mpnet-base-dot-v1':
        model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')  # max_seq_len: 512
    elif name == 'sgpt-125m':
        model = SentenceTransformer('Muennighoff/SGPT-125M-weightedmean-nli-bitfit')
    elif name == 'sgpt-1.3b':
        model = SentenceTransformer('Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit')
    elif name == 'sgpt-2.7b':
        model = SentenceTransformer('Muennighoff/SGPT-2.7B-weightedmean-nli-bitfit')
    elif name == 'sgpt-5.8b':
        model = SentenceTransformer('Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit')
    else:
        raise NotImplementedError

    dense_model = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=encoding_size,
                               activation_function=nn.Tanh())
    model = ContrastiveLearningST(modules=[model, dense_model, Normalize(p=2)])  # normalize the embedding
    return model


# Perform an L_p normalization on the input
class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        norm_sent_embeddings = F.normalize(x['sentence_embedding'], dim=1, p=self.p)
        x['sentence_embedding'] = norm_sent_embeddings
        return x
