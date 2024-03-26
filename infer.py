import os
import torch
import argparse
from tqdm import tqdm
from utils.misc import seed_everything
from data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import numpy as np
from utils.utilDistinct import ngrams
from simcse import SimCSE

def distinct_n_corpus_level(sentences, n):
    n_grams = []
    for sentence in sentences:
        if len(sentence) == 0:
            n_gram = []
        else:
            n_gram = list(ngrams(sentence, n) )  
        n_grams.extend(n_gram)

    length= len(n_grams) 
    distinct_ngrams = set(n_grams)
    if length == 0:
        return 0.0 
    return len(distinct_ngrams)/ length

def diversity_changed_score(ground, output, similarity):
    if len(output) == 0:
        return 0.0, 0.0  # Prevent a zero division
    ground_gram = set(ngrams(ground, 1))  
    output_gram = set(ngrams(output, 1))
    diff_list_1 = list(output_gram.difference(ground_gram))
    diff_list_2 = list(ground_gram.difference(output_gram))
    diverse_score = (len(diff_list_1) + len(diff_list_2))/ len(output_gram.union(ground_gram))
    diverse_similarty = diverse_score * np.exp(similarity - 1)
    return diverse_score, diverse_similarty

####Infer Stage#####
def train_infer_forward(args, iteration):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    logging.info("Create model.........")
    model_class, tokenizer_class = (BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    
    logging.info("Create train_loader.........")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'param_forward_WQ_' + str(iteration) +'.pt')) )    
   
    model = torch.nn.DataParallel(model, device_ids = [0, 1])
    model = model.to(device)
    test_nums_forward(model, tokenizer, args, args.return_nums, iteration) 
    
#####test_nums_forward() generate top-k questions########
def test_nums_forward(model, tokenizer, args, return_seq, iteration):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.module.to(device)
    model.eval()
    with torch.no_grad():
        outs = []
        sequences_likelihood = [] 
        test_pt = os.path.join(args.output_dir, 'forward_raw_test.pt')
        test_loader = DataLoader(test_pt, args.batch_size)
        for batch in tqdm(test_loader, total=len(test_loader)):            
            batch = batch[:2]
            source_ids, source_mask= [x.to(device) for x in batch]
            outputs = model.generate(
                input_ids = source_ids,                
                attention_mask = source_mask,
                max_length = 64,
                num_beams = 5,
                num_return_sequences = return_seq, 
                return_dict_in_generate=True, 
                output_scores = True
               )          
   
            for output_id, sequence_score in zip(outputs['sequences'], outputs['sequences_scores']):
                out = tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) 
                outs.append(out)
                sequences_likelihood.append(sequence_score.item())
    temp_outputs = []
    for line in outs:
        line = line.strip()
        pos = line.find("'s")    
        if pos !=-1 and pos >= 1:
            if line[pos-1]!= ' ':
                line = line.replace("'s", " 's")
        l = line.split(" ")
        ll = l[-1].split("?")
        if ll[0]!= "":
            l[-1] = ll[0]
            l.append("?")
            l = " ".join(l)
            l = " " if len(l.strip()) == 0 else l.strip()
            temp_outputs.append(l)
        else:
            line = " " if len(line.strip()) == 0 else line.strip()
            temp_outputs.append(line)
    outs = temp_outputs    
    print("semantic_score, diversity_score, bleu1, dist1_corpus:")
    print(evaluate_infer_multi_metrics(args.input_dir, args.output_dir, outs, return_seq))
   
#### infer evaluate semantic_score/diverse_score/BLEU/Dist
def evaluate_infer_multi_metrics(path_gold, output_dir, outs, k):
    golds = []
    predicts = []
    with open(os.path.join(path_gold, 'test_question_gold.txt')) as f:
        for line in f.readlines():
            golds.append(line.strip())
    predict = []
    for index, ele in enumerate(outs):
        predict.append(ele)            
        if (index+1) % k == 0:
            predicts.append(predict)
            predict = []
    similarities = [] 
    my_similar = [] 
    my_diverse = []
    similarities_filter = []
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    filter_preds = []
    bleu1_lists = []
    dist1_corpus_list = []
    for preds, ground in zip(predicts, golds):
        filter_pred = []
        bleu1_list = []
        sentence_b = []
        sentence_b.append(ground)
        similarity = model.similarity(sentence_b, preds) ###[[1,2,3,4,5]]
        similarities.append(np.mean(similarity[0]))
        my_similar.extend(similarity[0])
        ground = nltk.word_tokenize(ground)

        preds_list = []   
        for sim, pred in zip(similarity[0], preds):
            if sim >= 0.7:
                similarities_filter.append(sim)
                filter_pred.append(pred)
            pred = nltk.word_tokenize(pred)     
            my_diverse_score, _ = diversity_changed_score(ground, pred, 0) 
            my_diverse.append(my_diverse_score)    
            preds_list.append(pred)
            bleu1 = sentence_bleu([ground], pred, weights=(1.0, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
            bleu1_list.append(bleu1)
        dist_corpus_1 = distinct_n_corpus_level(preds_list, 1) 
        bleu1_lists.append(np.mean(bleu1_list))
        dist1_corpus_list.append(dist_corpus_1)
        filter_preds.append(filter_pred)       
        
        temp_diversity = []
        if len(filter_pred) > 0:
            for pred in filter_pred:
                pred = nltk.word_tokenize(pred)  
                diverse_score, _ = diversity_changed_score(ground, pred, 0)
                temp_diversity.append(diverse_score)
    diversity_scores = calculate_pair_preds_diverse(filter_preds)
    return  np.mean(similarities), np.mean(diversity_scores),np.mean(bleu1_lists), np.mean(dist1_corpus_list)

def calculate_pair_preds_diverse(preds):
    diversity_scores = []
    pred_length = []
    for pred in preds:
        true_diversity = []
        if len(pred) == 0:
            diversity_scores.append(0) 
            pred_length.append(0)
            continue
        if len(pred) == 1:
            diversity_scores.append(0)
            pred_length.append(0)
            continue
        for index, p in enumerate(pred):
            p = nltk.word_tokenize(p)  
            for j in range(index + 1, len(pred)):                
                p2 = nltk.word_tokenize(pred[j])
                diverse_score, _ = diversity_changed_score(p, p2, 0)
                true_diversity.append(diverse_score)
        diversity_scores.append(np.mean(true_diversity))
        pred_length.append(len(pred))
    return diversity_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--save_dir', required=False, help='path to save checkpoints and logs')
    parser.add_argument('--model_name_or_path', required = True)
    parser.add_argument('--return_nums', default=5, type=int, help='return sequence numbers')
    parser.add_argument('--filter_forward', default=True, help='whether filter train forward model data')
    parser.add_argument('--filter_backward', default=True, help='whether filter train backward model data')
    parser.add_argument('--N', default=5, type=int, help='iteration numbers')
    parser.add_argument('--percent', default=0.8, type=float, help='filter rate')
    parser.add_argument('--num_finetune_epochs', default=5, type = int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--learning_rate', default=3e-5, type = float)
    parser.add_argument('--num_train_epochs', default=30, type = int)
    parser.add_argument('--save_steps', default=448, type = int)
    parser.add_argument('--logging_steps', default=448, type = int)
    parser.add_argument('--warmup_proportion', default=0.1, type = float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default = 1e-4, type = float)
    args = parser.parse_args()
    ### args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))
    ### set random seed
    seed_everything(666)
    for iteration in range(args.N):
        train_infer_forward(args, iteration)
if __name__ == '__main__':
    main()
