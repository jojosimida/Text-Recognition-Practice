import numpy as np
import torch
from sklearn.metrics import accuracy_score
import string


class Evaluation():

    def __init__(self, model, loader):
        self.model = model
        self.loader = loader


    def evaluate(self):
        all_decodeGT = []
        all_decodePredict = []

        for images, texts in self.loader:
            self.model.eval() 
            output = self.model(images)

            decodeGT = self.decodetensor(texts.cpu().detach().numpy())
            decodePredict = self.decodetensor(self.argmax(output.cpu().detach().numpy()))

            all_decodeGT.extend(decodeGT)
            all_decodePredict.extend(decodePredict)

        acc = accuracy_score(all_decodeGT, all_decodePredict)
        return acc


    def decodetensor(self, tensors):
        decodedtexts = []
        table = str.maketrans(dict.fromkeys(string.punctuation)) 
        idx2char = self.model.gt.idx2char
        for tensor in tensors:
            decodedtext = []
            for t in tensor:
                if t!=self.model.gt.char2idx['<EOS>']:
                    decodedtext.extend(idx2char[t])
                else:
                    # decodedtexts.append(''.join(decodedtext).lower().translate(table))
                    decodedtexts.append(''.join(decodedtext).lower())
                    break
        return decodedtexts

    def argmax(self, output):
        return output.argmax(-1)

 

