import torch
from transformers import BertTokenizer, BertModel,BertForMaskedLM
from promptflow import tool
import random


def predict_seqs_dict(sequence, model, tokenizer, top_k=5, order="right-to-left"):  
                                                                                 
    ids_main = tokenizer.encode(sequence, return_tensors="pt", add_special_tokens=False)  
                                                                                 
    ids_ = ids_main.detach().clone()                                             
    position = torch.where(ids_main == tokenizer.mask_token_id)                  
                                                                                 
    positions_list = position[1].numpy().tolist()                                
                                                                                 
    if order == "left-to-right":                                                 
        positions_list.reverse()                                                 
                                                                                 
    elif order == "random":                                                      
        random.shuffle(positions_list)                                           
                                                                                 
    # print(positions_list)                                                      
    predictions_ids = {}                                                         
    predictions_detokenized_sents = {}                                           
                                                                                 
    for i in range(len(positions_list)):                                          
        predictions_ids[i] = []                                                   
        predictions_detokenized_sents[i] = []                                     
                                                                                  
        # if it was the first prediction,                                         
        # just go on and predict the first predictions                            
                                                                                  
        if i == 0:                                                                
            model_logits = model(ids_main)["logits"][0][positions_list[0]]           
            top_k_tokens = torch.topk(model_logits, top_k, dim=0).indices.tolist()  
                                                                                  
            for j in range(len(top_k_tokens)):                                    
                # print(j)                                                        
                ids_t_ = ids_.detach().clone()                                    
                ids_t_[0][positions_list[0]] = top_k_tokens[j]                    
                predictions_ids[i].append(ids_t_)                                 
                                                                                  
                pred = tokenizer.decode(ids_t_[0])                                
                predictions_detokenized_sents[i].append(pred)                     
                                                                                  
                # append the sentences and ids of this masked token               
                                                                                      
        # if we already have some predictions, go on and fill the rest of the masks  
        # by continuing the previous predictions                                                                                                                                                                                                                                          
        if i != 0:                                                                                                                                                                                                                                                                        
            for pred_ids in predictions_ids[i - 1]:                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                          
                # get the logits                                                                                                                                                                                                                                                          
                model_logits = model(pred_ids)["logits"][0][positions_list[i]]                                                                                                                                                                                                            
                # get the top 5 of this prediction and masked token                                                                                                                                                                                                                       
                top_k_tokens = torch.topk(model_logits, top_k, dim=0).indices.tolist()                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                          
                for top_id in top_k_tokens:                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                          
                    ids_t_i = pred_ids.detach().clone()                                                                                                                                                                                                                                   
                    ids_t_i[0][positions_list[i]] = top_id                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                          
                    pred = tokenizer.decode(ids_t_i[0])                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                          
                    # append the sentences and ids of this masked token                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                          
                    predictions_ids[i].append(ids_t_i)                                                                                                                                                                                                                                    
                    predictions_detokenized_sents[i].append(pred)                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                          
    return predictions_detokenized_sents   

tokenizer = BertTokenizer.from_pretrained('google-bert/bert-large-cased-whole-word-masking')
model = BertForMaskedLM.from_pretrained('google-bert/bert-large-cased-whole-word-masking')

@tool
def my_python_tool(prompt : str):
    return predict_seqs_dict(prompt, model, tokenizer)
