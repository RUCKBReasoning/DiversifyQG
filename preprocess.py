import os
import torch
import json
import pickle
import argparse
import numpy as np
import nltk
from tqdm import tqdm
import sys
import heapq
from transformers import BartTokenizer
from simcse import SimCSE
from utils.utilDistinct import ngrams
from data import DataLoader

def diversity_changed_score(ground, output):
    if len(output) == 0:
        return 0.0  # Prevent a zero division
    ground_gram = set(ngrams(ground, 1))  
    output_gram = set(ngrams(output, 1))
    diff_list_1 = list(output_gram.difference(ground_gram))
    diff_list_2 = list(ground_gram.difference(output_gram))
    diverse_score = (len(diff_list_1) + len(diff_list_2))/ len(output_gram.union(ground_gram))
    return diverse_score

def getNovelSubgraph(subkg):
    g_nodes = subkg['g_node_names']
    g_adj = subkg['g_adj'] ### type is a dict
    seq = []    
    for key, value in g_adj.items():
        subject = g_nodes[key]
        if subject == "none":
            continue
        else:
            for k in list(value.keys()):
                obj = g_nodes[k]
                if obj == "none" and k in list(g_adj.keys()):
                    value.update(g_adj[k])
                    
    for key, value in g_adj.items():
        subject = g_nodes[key]
        if subject == "none":
            continue
        else:
            for k, relation in value.items():
                obj = g_nodes[k]
                if obj == "none":
                    continue
                else:
                    relation = relation.strip().split('/')[-1]
                    if relation.find('_')!=-1:
                        relation = relation.split('_')
                        relation = ' '.join(relation).strip()
                    fact = "{} {} {}".format(subject, relation, obj)
                    seq.append(fact)
    subkg = ' </s> '.join(seq)  
                  
    return subkg

def get_subkg_seq(subkg): 
    seq = []
    maskList = []
    g_nodes = subkg['g_node_names']
    g_edges = subkg['g_edge_types']
    g_adj = subkg['g_adj']
    all_subjects = []
    all_objects = []
    for key, value in g_adj.items():
        subject = g_nodes[key]
        all_subjects.append(subject)
        for k, relation in value.items():
            obj = g_nodes[k]
            all_objects.append(obj)
            #### PQ relation is a list ####
            # relation = relation[0]
            # if relation.find('/') >= 0:
            #     relation = relation.strip().split('/')[-1]
            #### WQ relation is a str ####
            relation = relation.strip().split('/')[-1]
            if relation.find('_')!=-1:
                relation = relation.split('_')
                relation = ' '.join(relation).strip()
            fact = "{} {} {}".format(subject, relation, obj)
            seq.append(fact)
    subkg = ' </s> '.join(seq)
    return subkg, seq, maskList 

def encode_dataset(dataset, tokenizer, test):
    max_seq_length = 512
    questions = []
    answers = []
    subkgs = []
    for item in tqdm(dataset):
        question = item['outSeq']
        questions.append(question)
        subkg = item['inGraph']
        subkg = getNovelSubgraph(subkg)
        subkgs.append(subkg)
        answer = item['answers']       
        if len(answer) > 1:
            answer = [', '.join(answer)]
        if len(answer) == 0:
            answer = ['']
        answers = answers + answer
    s = [i +' </s> ' + j for i, j in zip(subkgs, answers)] 
    
    input_ids = tokenizer.batch_encode_plus(s, max_length = max_seq_length, padding = "max_length", truncation = True)
    source_ids = np.array(input_ids['input_ids'], dtype = np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype = np.int32)

    if not test:
        target_ids = tokenizer.batch_encode_plus(questions, max_length = max_seq_length, padding = "max_length", truncation = True)
        target_ids = np.array(target_ids['input_ids'], dtype = np.int32)
    else:
        target_ids = np.array([], dtype = np.int32)
 
    return source_ids, source_mask, target_ids


def encode_back_dataset(dataset, tokenizer, test):
    max_seq_length = 512
    questions = []
    subkgs = []
    for item in tqdm(dataset):
        question = item['outSeq']
        questions.append(question)
        subkg = item['inGraph']
        subkg = getNovelSubgraph(subkg)
        subkgs.append(subkg)
       
    input_ids = tokenizer.batch_encode_plus(questions, max_length = max_seq_length, padding = "max_length", truncation = True)
    source_ids = np.array(input_ids['input_ids'], dtype = np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype = np.int32)

    if not test:
        target_ids = tokenizer.batch_encode_plus(subkgs, max_length = max_seq_length, padding = "max_length", truncation = True)
        target_ids = np.array(target_ids['input_ids'], dtype = np.int32)
    else:
        target_ids = np.array([], dtype = np.int32)
 
    return source_ids, source_mask, target_ids

def encode_question(questions, tokenizer):
    max_seq_length = 64    
    input_ids = tokenizer.batch_encode_plus(questions, max_length = max_seq_length, padding = "max_length", truncation = True)
    source_ids = np.array(input_ids['input_ids'], dtype = np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype = np.int32)
    target_ids = np.array([], dtype = np.int32) 
    return source_ids, source_mask, target_ids   

def encode_pseudo_train(outs, y, tokenizer):
    max_seq_length = 512
    input_ids = tokenizer.batch_encode_plus(outs, max_length = max_seq_length, padding = "max_length", truncation = True)
    source_ids = np.array(input_ids['input_ids'], dtype = np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype = np.int32)
    target_ids = tokenizer.batch_encode_plus(y, max_length = 64, padding = "max_length", truncation = True)
    target_ids = np.array(target_ids['input_ids'], dtype = np.int32)
    return source_ids, source_mask, target_ids  


def encode_filter_one_dataset(subkgs, tokenizer):
    max_seq_length = 512
    input_ids = tokenizer.batch_encode_plus(subkgs, max_length = max_seq_length, padding = "max_length", truncation = True)
    source_ids = np.array(input_ids['input_ids'], dtype = np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype = np.int32)
    target_ids = np.array([], dtype = np.int32)
 
    return source_ids, source_mask, target_ids

def test_filter_one(model, device, tokenizer, args, name):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        sequences_likelihood = [] 
        outs = []
        test_pt = os.path.join(args.output_dir, name + '.pt')
        test_loader = DataLoader(test_pt, args.batch_size)
        for batch in tqdm(test_loader, total=len(test_loader)):            
            batch = batch[:2]
            source_ids, source_mask= [x.to(device) for x in batch]
            outputs = model.generate(
                input_ids = source_ids,
                attention_mask = source_mask,
                num_beams = args.return_nums,
                max_length = 64,
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
        if pos != -1 and pos >= 1:
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
    outs = temp_outputs ### outs_back generate questions
    return outs

def filter_forward(args, model, tokenizer, outs_back, device, y, iteration):
    ##### filter one strategy：2 steps ######     
    ##### Step 1: outs input model, generate questions
    outputs = encode_filter_one_dataset(outs_back, tokenizer)
    name = 'forward_filter_one_' + str(iteration)
    with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
        for o in outputs:
            print(o.shape)
            pickle.dump(o, f)
    
    outs_questions = test_filter_one(model, device, tokenizer, args, name)  ####### outs_back generate questions
    my_outs = []
    questions = []
    simi_list = []
    ##### Step 2: outs_questions compare with y according to similarity score
    sim_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    for index, ground  in enumerate(y):
        sentence_a = []
        sentence_b = []
        sentence_b.append(ground)
        outs_q = outs_questions[args.return_nums*index:args.return_nums*(index+1)]
        sentence_a.extend(outs_q)
        similarity = sim_model.similarity(sentence_b, sentence_a) ###[[1,2,3,4,5]]
        simi_list.extend(similarity[0])

        for sim, out_back in zip(similarity[0], outs_back[args.return_nums*index:args.return_nums*(index+1)]):
            if sim >= 0.7:
                my_outs.append(out_back)
                questions.append(ground)   
       
    return my_outs, questions    


def forward_process(args, tokenizer, outs_back, model, device, iteration): #### model is the forward model, outs_back is the subgraph of natural questions by backward model
    print('Loading!!!!')
    with open(os.path.join(args.input_dir, 'pseundo.txt'), 'r') as f:
        y = f.readlines()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    ####filter strategy###
    if(args.filter_forward):
        outs, questions = filter_forward(args, model, tokenizer, outs_back, device, y, iteration)
    else:
        questions = []
        for ground in y:
            questions.extend([ground for i in range(args.return_nums)])
        outs = outs_back

    #####construct training dataset####
    outputs = encode_pseudo_train(outs, questions, tokenizer)
    name = 'forward_train_' + str(iteration)
    with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
        for o in outputs:
            print(o.shape)
            pickle.dump(o, f)

##### prepare data for forward model
def generate_forward_process(args, tokenizer, iteration):
    train_set = []
    with open(os.path.join(args.input_dir, 'train.json')) as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            train_set.append(line) 

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    outputs = encode_dataset(train_set, tokenizer, True)
    name = 'forward_generate_' + str(iteration)
    print('shape of source_ids, source_mask, target_ids:')
    with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
        for o in outputs:
            print(o.shape)
            pickle.dump(o, f)


def backward_process(args, tokenizer, outs, iteration):
    print('Loading!!!!')   
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    groundTruth = []
    subkgs = []
    with open(os.path.join(args.input_dir, 'train.json')) as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            question = line['outSeq']
            groundTruth.append(question)
            subkg = line['inGraph']
            subkg = getNovelSubgraph(subkg)
            subkgs.append(subkg)
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
    with open(os.path.join(args.output_dir, 'filter_before_backward_finetune_length.txt'), 'a') as f:
        f.write("before raw length: " + str(len(outs)) + '\n')
    
    ####filter strategy###
    if(args.filter_backward):
        temp_outs = []
        temp_sequences = []
        temp_diverses = []
        sim_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        for index, ground  in enumerate(groundTruth): #### question's groundTruth
            sentence_a = []
            sentence_b = []
            sentence_b.append(ground)
            outs_q = outs[args.return_nums*index:args.return_nums*(index+1)]
            sentence_a.extend(outs_q)
            similarity = sim_model.similarity(sentence_b, sentence_a) ###[[1,2,3,4,5]]   
            grou = nltk.word_tokenize(ground)     
            for sim, q, sub in zip(similarity[0], outs_q, subkgs):             
                if sim >= 0.7: #### set the similarity threshold as 0.7
                    temp_outs.append(q)
                    temp_sequences.append(sub)
                    q = nltk.word_tokenize(q)  
                    diverse_score = diversity_changed_score(grou, q) #### our designed diversity score
                    temp_diverses.append(diverse_score)
        
        my_length = int(len(temp_diverses)*args.percent)
        my_subkgs = []
        my_outs = []
        index_max = heapq.nlargest(my_length, range(len(temp_diverses)), temp_diverses.__getitem__)
        for i in index_max:
            my_subkgs.append(temp_sequences[i])
            my_outs.append(temp_outs[i])
        subkgs = my_subkgs
        outs = my_outs
        with open(os.path.join(args.output_dir, 'filter_after_backward_finetune_length_' + str(iteration) +'.txt'), 'a') as f:
            f.write("after length: " + str(len(outs)) + '\n')
        
    else:
        sequences = []
        for sub in subkgs:
            sequences.extend([sub for i in range(args.return_nums)])
        subkgs = sequences

    #####construct training dataset#####
    outputs = encode_pseudo_train(outs, subkgs, tokenizer)
    name = 'backward_train_' + str(iteration)
    with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
        for o in outputs:
            print(o.shape)
            pickle.dump(o, f)

#####prepare data for backward ####
def generate_backward_process(args, tokenizer, iteration):
    print('Loading!!!!')
    train_set = []
    ##### pseundo.txt is unlabel data, form is questions####
    with open(os.path.join(args.input_dir, 'pseundo.txt'), 'r') as f:
        train_set = f.readlines()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    outputs = encode_question(train_set, tokenizer)
    name = 'backward_generate_' + str(iteration)
    print('shape of source_ids, source_mask, target_ids:')
    with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
        for o in outputs:
            print(o.shape)
            pickle.dump(o, f)


def forward_initialize_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    args = parser.parse_args()
    test_set = []
    train_set = []
    val_set = []
    
    with open(os.path.join(args.input_dir, 'test.json')) as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            test_set.append(line)
    with open(os.path.join(args.input_dir, 'train.json')) as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            train_set.append(line)
    with open(os.path.join(args.input_dir, 'dev.json')) as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            val_set.append(line)
   
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)        
           
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
   
    
    for name, dataset in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
        print('Encode {} set'.format(name))
        outputs = encode_dataset(dataset, tokenizer, name == 'test')
        name = 'forward_raw_' + name
        print('shape of source_ids, source_mask, target_ids:')  
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)

def backward_initialize_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    args = parser.parse_args()
    test_set = []
    train_set = []
    val_set = []
    
    with open(os.path.join(args.input_dir, 'test.json')) as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            test_set.append(line)
    with open(os.path.join(args.input_dir, 'train.json')) as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            train_set.append(line)
    with open(os.path.join(args.input_dir, 'dev.json')) as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            val_set.append(line)
   
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
   
    for name, dataset in zip(('train',  'val', 'test'), (train_set, val_set, test_set)):
        print('Encode {} set'.format(name))
        outputs = encode_back_dataset(dataset,tokenizer, name == 'test')
        print(type(outputs))
        name = 'backward_raw_' + name
        print('shape of source_ids, source_mask, target_ids:')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)

if __name__ == '__main__':
    forward_initialize_main()
    backward_initialize_main()
