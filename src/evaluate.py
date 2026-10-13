import time, numpy as np, torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def evaluate(model_id, texts, labels, batch_size=16, max_length=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    # Map model class ids to IMDB convention: negative=0, positive=1.
    id2label = {int(k):str(v).lower() for k,v in model.config.id2label.items()}
    def positive_index():
        for idx,name in id2label.items():
            if name in ('positive','pos','label_1') or 'positive' in name: return idx
        raise ValueError(f'Cannot verify positive label mapping for {model_id}: {id2label}')
    pos_idx=positive_index()
    def predict(batch):
        inputs=tokenizer(batch,padding=True,truncation=True,max_length=max_length,return_tensors='pt').to(device)
        with torch.inference_mode(): logits=model(**inputs).logits
        return (logits.argmax(-1)==pos_idx).long().cpu().tolist()
    # warm-up excluded from measured latency
    predict(texts[:min(batch_size,len(texts))])
    if device.type=='cuda': torch.cuda.synchronize()
    start=time.perf_counter(); predictions=[]
    for i in range(0,len(texts),batch_size): predictions.extend(predict(texts[i:i+batch_size]))
    if device.type=='cuda': torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    precision,recall,f1,_=precision_recall_fscore_support(labels,predictions,average='binary',zero_division=0)
    result={'model':model_id,'accuracy':accuracy_score(labels,predictions),'precision':precision,'recall':recall,'f1':f1,'total_seconds':seconds,'ms_per_review':seconds*1000/len(texts),'reviews_per_second':len(texts)/seconds,'gpu':str(device),'batch_size':batch_size,'max_length':max_length,'n':len(texts),'confusion_matrix':confusion_matrix(labels,predictions,labels=[0,1]).tolist()}
    del model,tokenizer
    if device.type=='cuda': torch.cuda.empty_cache()
    return result
