from sentence_transformers import SentenceTransformer, util

class ZeroShotTextClassifier:
    def __init__(self, language='pt', model_ref=None):
        if model_ref is not None:
            self.model = SentenceTransformer(model_ref)
        elif language=='en':
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        elif language=='pt':
            self.model = SentenceTransformer('rufimelo/bert-large-portuguese-cased-sts')
    
    def predict(self, X, labels, return_type='labels'):
        embedding_1= self.model.encode(X, convert_to_tensor=True)
        embedding_2= self.model.encode(labels, convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(embedding_1, embedding_2)
        idxs, percents = cos_sim.max(axis=1).indices, cos_sim.max(axis=1).values
        if return_type=='labels_with_percent':
            text_labels = []
            for i, percent in zip(idxs, percents):
                tl = f'{labels[i]} ({percent*100:.2f}%)'
                text_labels.append(tl)
            return text_labels
        if return_type=='labels':
            pred_labels = [labels[i] for i in idxs]
            return pred_labels
        else:
            return idxs, percents

