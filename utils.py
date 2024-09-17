import re
import spacy
import torch
from torch.autograd import Variable

import en_core_web_sm
NLP = en_core_web_sm.load()



def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def detach(x):
    """Detaches hidden states from their history to prevent backpropagation through time."""
    if isinstance(x, torch.Tensor):
        # Directly detach if x is a tensor
        return x.detach()
    elif isinstance(x, (list, tuple)):
        # If x is a list or tuple, recursively detach each element
        return type(x)(detach(v) for v in x)
    else:
        raise ValueError(f"Unexpected type for detach: {type(x)}")

    
    
def tokenizer(text):
    text = re.sub(b'\u200c'.decode("utf-8", "strict"), " ", text)   # replace half-spaces with spaces
    text = re.sub('\n', ' ', text)
    text = re.sub('-', ' - ', text)
    text = re.sub('[ ]+', ' ', text)
    text = re.sub('\.', ' .', text)
    text = re.sub('\،', ' ،', text)
    text = re.sub('\؛', ' ؛', text)
    text = re.sub('\؟', ' ؟', text)
    text = re.sub('\. \. \.', '...', text)
    
    return [w.text for w in NLP.tokenizer(str(text))]