import os
import torch
import argparse
import json
from tqdm import tqdm
from utils.misc import seed_everything
from data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.optim import AdamW
import logging
import time
import nltk
import numpy as np
from utils.utilDistinct import ngrams
from preprocess import getNovelSubgraph, forward_process, generate_forward_process, backward_process, generate_backward_process
from earlyStop import EarlyStopping
from simcse import SimCSE

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

def train_forward(args, iteration):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    logging.info("Create model.........")
    #### BART model
    model_class, tokenizer_class = (BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    
    logging.info("Create train_loader.........")
    if iteration == 0:
        train_pt = os.path.join(args.output_dir, 'forward_raw_train.pt')
        train_loader = DataLoader(train_pt, args.batch_size, training=True)
    else:             
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'param_forward_WQ_' + str(iteration - 1) +'.pt')) )
        backward_model = model_class.from_pretrained(args.model_name_or_path)
        backward_model.load_state_dict(torch.load(os.path.join(args.output_dir, 'param_backward_WQ_' + str(iteration - 1) +'.pt')) )
        outs = generate_backward(backward_model, tokenizer, device, iteration, args)  #### backward model generate subgraphs of natural questions      
        forward_process(args, tokenizer, outs, model, device, iteration)
        train_pt = os.path.join(args.output_dir, 'forward_train_' + str(iteration) + '.pt')
        train_loader = DataLoader(train_pt, args.batch_size, training=True)
    

    model = torch.nn.DataParallel(model, device_ids = [0, 1])
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)   
    start = time.time()
    #### Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_loader.dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)  
    earlyStop = EarlyStopping()  
    optimizer.zero_grad()
    semanticMaxScore = 0.0
    name = 'forward'
    for i in range(args.num_train_epochs):
        model.train()
        global_step = 0
        tr_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            pad_token_id = tokenizer.pad_token_id
            source_ids, source_mask, y = batch[0], batch[1], batch[2]            
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()            
            lm_labels[y[:, 1:] == pad_token_id] = -100      
            outputs = model(input_ids = source_ids.to(device), attention_mask = source_mask.to(device), decoder_input_ids = y_ids.to(device), labels = lm_labels.to(device))
            loss = outputs[0]           
            loss.mean().backward()
            print("loss:",loss.mean().item())
            tr_loss += loss.mean().item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        epoch_loss = tr_loss/global_step ### after each epoch, the loss value #####
        earlyStop(epoch_loss, model, args, name, iteration)   
        if 'cuda' in str(device):
            torch.cuda.empty_cache()        
        out, semantic_score, diversity_score, _ = test_forward(model, tokenizer, args, iteration)  
        sim_div = diversity_score*np.exp(semantic_score - 1)                  
        if semanticMaxScore < sim_div: 
            semanticMaxScore = sim_div
            torch.save(model.module.state_dict(), os.path.join(args.output_dir, 'param_forward_WQ_' + str(iteration) +'.pt'), _use_new_zipfile_serialization = False)
            np.savetxt(os.path.join(args.output_dir, 'forward_WQ_Val_' + str(iteration) +'.txt'), out, fmt='%s')
        if(earlyStop.early_stop): ####early stop#####
            break
    
    
    #### if iteration>=1, then finetune####
    if iteration >= 1:
        train_pt = os.path.join(args.output_dir, 'forward_raw_train.pt')
        train_loader = DataLoader(train_pt, args.batch_size, training=True)
        for i in range(args.num_finetune_epochs):
            model.train()
            global_step = 0
            tr_loss = 0.0
            for step, batch in enumerate(train_loader):
               
                batch = tuple(t.to(device) for t in batch)
                pad_token_id = tokenizer.pad_token_id
                source_ids, source_mask, y = batch[0], batch[1], batch[2]
                
                y_ids = y[:, :-1].contiguous()
                lm_labels = y[:, 1:].clone()
                
                lm_labels[y[:, 1:] == pad_token_id] = -100      
                outputs = model(input_ids = source_ids.to(device), attention_mask = source_mask.to(device), decoder_input_ids = y_ids.to(device), labels = lm_labels.to(device))
                loss = outputs[0]           
                loss.mean().backward()
                print("loss:",loss.mean().item())
                tr_loss += loss.mean().item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            epoch_loss = tr_loss/global_step 
            earlyStop(epoch_loss, model, args, name, iteration)   
            if 'cuda' in str(device):
                torch.cuda.empty_cache()
            
            out, semantic_score, diversity_score, _ = test_forward(model, tokenizer, args, iteration)  
            sim_div = diversity_score*np.exp(semantic_score - 1)                  
            if semanticMaxScore < sim_div: 
                semanticMaxScore = sim_div
                torch.save(model.module.state_dict(), os.path.join(args.output_dir, 'param_forward_WQ_' + str(iteration) +'.pt'), _use_new_zipfile_serialization = False)
                np.savetxt(os.path.join(args.output_dir, 'forward_WQ_Val_Finetune_' + str(iteration) +'.txt'), out, fmt='%s')
            if(earlyStop.early_stop): 
                break
    
    
    end = time.time()
    t = (end-start)/3600
    print("run time: {} h".format(t))


def train_backward(args, iteration):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    logging.info("Create model.........")
    #### BART model
    model_class, tokenizer_class = (BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    
    logging.info("Create train_loader.........")
    if iteration == 0:
        train_pt = os.path.join(args.output_dir, 'backward_raw_train.pt')
        train_loader = DataLoader(train_pt, args.batch_size, training=True)
    else:
        ##### using the forward model get y about label data, then pass backward_process, finally obtain data for backward model#######
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'param_backward_WQ_' + str(iteration - 1) +'.pt')) )
        forward_model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
        forward_model.load_state_dict(torch.load(os.path.join(args.output_dir, 'param_forward_WQ_' + str(iteration - 1) +'.pt')) )
        outs = generate_forward(forward_model, tokenizer, device, iteration, args)        
        backward_process(args, tokenizer, outs, iteration)
        train_pt = os.path.join(args.output_dir, 'backward_train_' + str(iteration) + '.pt')
        train_loader = DataLoader(train_pt, args.batch_size, training=True)  

    model = torch.nn.DataParallel(model, device_ids = [0, 1])
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
   
    start = time.time()
    ### Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_loader.dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)  
    earlyStop = EarlyStopping()  
    optimizer.zero_grad()
    semanticMaxScore = 0.0
    name = 'backward'
    for i in range(int(args.num_train_epochs)):
        model.train()
        global_step = 0
        tr_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            pad_token_id = tokenizer.pad_token_id
            source_ids, source_mask, y = batch[0], batch[1], batch[2]            
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()            
            lm_labels[y[:, 1:] == pad_token_id] = -100      
            outputs = model(input_ids = source_ids.to(device), attention_mask = source_mask.to(device), decoder_input_ids = y_ids.to(device), labels = lm_labels.to(device))
            loss = outputs[0]           
            loss.mean().backward()
            print("loss:",loss.mean().item())
            tr_loss += loss.mean().item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        epoch_loss = tr_loss/global_step ### after each epoch, the loss value###
        earlyStop(epoch_loss, model, args, name, iteration)   
        if 'cuda' in str(device):
            torch.cuda.empty_cache()
        
        semantic_score, outs = test_backward(model, tokenizer, args, iteration)                    
        if semanticMaxScore < semantic_score : 
            semanticMaxScore = semantic_score
            torch.save(model.module.state_dict(), os.path.join(args.output_dir, 'param_backward_WQ_' + str(iteration) +'.pt'), _use_new_zipfile_serialization = False)
            np.savetxt(os.path.join(args.output_dir, 'backward_WQ_Val_' + str(iteration) +'.txt'), outs, fmt='%s')
        if(earlyStop.early_stop): ####early stop#####
            break
    
    #### if iteration>=1, then finetune####
    if iteration >= 1:
        train_pt = os.path.join(args.output_dir, 'backward_raw_train.pt')
        train_loader = DataLoader(train_pt, args.batch_size, training=True)
        for i in range(args.num_finetune_epochs):
            model.train()
            global_step = 0
            tr_loss = 0.0
            for step, batch in enumerate(train_loader):
                
                batch = tuple(t.to(device) for t in batch)
                pad_token_id = tokenizer.pad_token_id
                source_ids, source_mask, y = batch[0], batch[1], batch[2]
                
                y_ids = y[:, :-1].contiguous()
                lm_labels = y[:, 1:].clone()
                
                lm_labels[y[:, 1:] == pad_token_id] = -100      
                outputs = model(input_ids = source_ids.to(device), attention_mask = source_mask.to(device), decoder_input_ids = y_ids.to(device), labels = lm_labels.to(device))
                loss = outputs[0]           
                loss.mean().backward()
                print("loss:",loss.mean().item())
                tr_loss += loss.mean().item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            epoch_loss = tr_loss/global_step ### after each epoch, the loss value
            earlyStop(epoch_loss, model, args, name, iteration)   
            if 'cuda' in str(device):
                torch.cuda.empty_cache()
            
            semantic_score, outs = test_backward(model, tokenizer, args, iteration)                    
            if semanticMaxScore < semantic_score : 
                semanticMaxScore = semantic_score
                torch.save(model.module.state_dict(), os.path.join(args.output_dir, 'param_backward_WQ_' + str(iteration) +'.pt'), _use_new_zipfile_serialization = False)
                np.savetxt(os.path.join(args.output_dir, 'backward_WQ_Val_Finetune_' + str(iteration) +'.txt'), outs, fmt='%s')
            if(earlyStop.early_stop):
                break    
    
    
    end = time.time()
    t = (end-start)/3600
    print("run time: {} h".format(t))

###### test_forward() initialize for training the forward function######
def test_forward(model, tokenizer, args, iteration):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.module.to(device)
    model.eval()
    with torch.no_grad():
        sequences_likelihood = [] 
        outs = []
        test_pt = os.path.join(args.output_dir, 'forward_raw_val.pt')
        test_loader = DataLoader(test_pt, args.batch_size)
        for batch in tqdm(test_loader, total=len(test_loader)):            
            batch = batch[:2]
            source_ids, source_mask= [x.to(device) for x in batch]
            outputs = model.generate(
                input_ids = source_ids,
                attention_mask = source_mask,
                num_beams = args.return_nums,
                max_length = 64,
                num_return_sequences = args.return_nums, 
                early_stopping=True,
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
    semantic_similarity, diversity_score, diversity_ground_score = evaluate_multi_metrics(args.input_dir, outs, args.return_nums)
   
    out = np.array(outs)
    return out, semantic_similarity, diversity_score, diversity_ground_score

#####generate_forward() aims to prepare training data for training backward function ######
def generate_forward(model, tokenizer, device, iteration, args):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        sequences_likelihood = [] 
        outs = []
        generate_forward_process(args, tokenizer, iteration)
        test_pt = os.path.join(args.output_dir, 'forward_generate_' + str(iteration) +'.pt')
        test_loader = DataLoader(test_pt, args.batch_size)
        for batch in tqdm(test_loader, total=len(test_loader)):            
            batch = batch[:2]
            source_ids, source_mask= [x.to(device) for x in batch]
            outputs = model.generate(
                input_ids = source_ids,
                attention_mask = source_mask,
                num_beams = args.return_nums,
                max_length = 64,
                num_return_sequences = args.return_nums, 
                early_stopping=True,
                return_dict_in_generate=True, 
                output_scores = True
            )
            for output_id, sequence_score in zip(outputs['sequences'], outputs['sequences_scores']):
                out = tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) 
                outs.append(out)
                sequences_likelihood.append(sequence_score.item())
    temp_outputs = []
    for line in outs:
        l = line.strip().split(" ")
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
    out = np.array(outs)
    np.savetxt(os.path.join(args.output_dir, 'predict_forward_WQ_' + str(iteration) +'.txt'), out, fmt='%s')
    return out

######test_backward() is used to initialize the forward function######
def test_backward(model, tokenizer, args, iteration):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.module.to(device)
    model.eval()
    with torch.no_grad():
        sequences_likelihood = [] 
        outs = []
        test_pt = os.path.join(args.output_dir, 'backward_raw_val.pt')
        test_loader = DataLoader(test_pt, args.batch_size)
        for batch in tqdm(test_loader, total=len(test_loader)):            
            batch = batch[:2]
            source_ids, source_mask= [x.to(device) for x in batch]
            outputs = model.generate(
                input_ids = source_ids,
                attention_mask = source_mask,
                num_beams = args.return_nums,
                max_length = 512,
                num_return_sequences = args.return_nums, 
                early_stopping=True,
                return_dict_in_generate=True, 
                output_scores = True
            )
            for output_id, sequence_score in zip(outputs['sequences'], outputs['sequences_scores']):
                out = tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) 
                outs.append(out)
                sequences_likelihood.append(sequence_score.item())
            
    groundTruth = []
    with open(os.path.join(args.input_dir, 'dev.json')) as f:
        for line in f.readlines():
            line = line.strip()
            line = json.loads(line)
            subkg = line['inGraph']
            subkg = getNovelSubgraph(subkg)
            groundTruth.append(subkg)
 
    ####calculate semantic score ####
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    semantics = []
    for index, ground  in enumerate(groundTruth):  
        sentence_a = []
        sentence_b = []
        sentence_a.extend(outs[index*args.return_nums:(index+1)*args.return_nums])
        sentence_b.append(ground)
        similarity = model.similarity(sentence_b, sentence_a) ###[[1,2,3,4,5]]
        semantics.append(np.mean(similarity[0]))
    with open(os.path.join(args.output_dir, 'out_score_backward_WQ_' + str(iteration) +'.txt'), 'a') as f:
        f.write(str(np.mean(semantics)) + '\n')
    return np.mean(semantics), outs

####generate_backward() aims to prepare the training data for training forward model######
def generate_backward(model, tokenizer, device, iteration, args):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        sequences_likelihood = [] 
        outs = []
        generate_backward_process(args, tokenizer, iteration)
        test_pt = os.path.join(args.output_dir, 'backward_generate_' + str(iteration) +'.pt')
        test_loader = DataLoader(test_pt, args.batch_size)
        for batch in tqdm(test_loader, total=len(test_loader)):            
            batch = batch[:2]
            source_ids, source_mask= [x.to(device) for x in batch]
            outputs = model.generate(
                input_ids = source_ids,
                attention_mask = source_mask,
                num_beams = args.return_nums,
                max_length = 512,
                num_return_sequences = args.return_nums, 
                early_stopping=True,
                return_dict_in_generate=True, 
                output_scores = True
            )
            for output_id, sequence_score in zip(outputs['sequences'], outputs['sequences_scores']):
                out = tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) 
                outs.append(out)
                sequences_likelihood.append(sequence_score.item())

    np.savetxt(os.path.join(args.output_dir, 'predict_backward_WQ_' + str(iteration) +'.txt'), outs, fmt='%s')
    return outs


def evaluate_multi_metrics(path_gold, outs, k):
    golds = []
    predicts = []
    with open(os.path.join(path_gold, 'val_question_gold.txt')) as f:
        for line in f.readlines():
            golds.append(line.strip())
    predict = []
    for index, ele in enumerate(outs):
        predict.append(ele)            
        if (index+1) % k == 0:
            predicts.append(predict)
            predict = []
    similarities = [] 
    similarities_filter = []
    diversity_ground_scores = [] 
    diversity_scores = []
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    filter_preds = []
    for preds, ground in zip(predicts, golds):
        filter_pred = []
        sentence_b = []
        sentence_b.append(ground)
        similarity = model.similarity(sentence_b, preds) ###[[1,2,3,4,5]]
        similarities.append(np.mean(similarity[0]))
        for sim, pred in zip(similarity[0], preds):
            if sim >= 0.7:
                similarities_filter.append(sim)
                filter_pred.append(pred)
        filter_preds.append(filter_pred)

        temp_diversity = []
        if len(filter_pred) > 0:
            ground = nltk.word_tokenize(ground)
            for pred in filter_pred:
                pred = nltk.word_tokenize(pred)  
                diverse_score, _ = diversity_changed_score(ground, pred, 0)
                diverse_score = round(diverse_score, 4)
                temp_diversity.append(diverse_score)
            diversity_ground_scores.append(np.mean(temp_diversity))
    
    for pred in filter_preds:
        true_diversity = []
        if len(pred) == 0:
            diversity_scores.append(0)
            continue
        if len(pred) == 1:
            diversity_scores.append(0)
            continue
        for index, p in enumerate(pred):
            p = nltk.word_tokenize(p)  
            for j in range(index + 1, len(pred)):                
                p2 = nltk.word_tokenize(pred[j])
                diverse_score, _ = diversity_changed_score(p, p2, 0)
                true_diversity.append(diverse_score)
        diversity_scores.append(np.mean(true_diversity))
    return  np.mean(similarities), np.mean(diversity_scores), np.mean(diversity_ground_scores)


def main():
    parser = argparse.ArgumentParser()
    #### input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--save_dir', required=False, help='path to save checkpoints and logs')
    parser.add_argument('--model_name_or_path', required = True)
    parser.add_argument('--ckpt')

    #### training parameters
    parser.add_argument('--return_nums', default=5, type=int, help='return sequence numbers')
    parser.add_argument('--filter_forward', default=True, help='whether filter train forward model data')
    parser.add_argument('--filter_backward', default=True, help='whether filter train backward model data')
    parser.add_argument('--N', default=3, type=int, help='iteration numbers, WQ is 3 and PQ is 2')
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
    #### args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))
    #### set seed
    seed_everything(666)
    for iteration in range(args.N):
        train_forward(args, iteration)
        train_backward(args, iteration)       
    
if __name__ == '__main__':
    main()
   
    
