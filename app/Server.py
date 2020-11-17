from flask import Flask, request, session, render_template
import pandas as pd
from unidecode import unidecode
import re
import json
import os
from subprocess import Popen, PIPE, STDOUT
from elasticsearch import Elasticsearch
from time import sleep
import collections

from transformers import BasicTokenizer ,AutoTokenizer, AutoModelForQuestionAnswering
import torch

from tqdm.notebook import tqdm

def populate_index(es_obj, index_name, evidence_corpus):
    '''
    Loads records into an existing Elasticsearch index

    Args:
        es_obj (elasticsearch.client.Elasticsearch) - Elasticsearch client object
        index_name (str) - Name of index
        evidence_corpus (list) - List of dicts containing data records

    '''

    for i, rec in enumerate(tqdm(evidence_corpus)):
    
        try:
            index_status = es_obj.index(index=index_name, id=i, body=rec)
        except:
            print(f'Unable to load document {i}.')
            
    n_records = es_obj.count(index=index_name)['count']
    print(f'Succesfully loaded {n_records} into {index_name}')


    return

def parse_qa_records(data):
    '''
    Loop through SQuAD2.0 dataset and parse out question/answer examples and unique article paragraphs
    
    Returns:
        qa_records (list) - Question/answer examples as list of dictionaries
        documents (list) - Unique titles and article paragraphs recreated from SQuAD data
    
    '''
    num_with_ans = 0
    num_without_ans = 0
    qa_records = []
    documents = {}
    
    for article in data:
        
        for i, paragraph in enumerate(article['paragraphs']):
            
            documents[article['title']+f'_{i}'] = article['title'] + ' ' + paragraph['context']
            
            for questions in paragraph['qas']:
                
                qa_record = {}
                qa_record['example_id'] = questions['id']
                qa_record['document_title'] = article['title']
                qa_record['question_text'] = questions['question']
                
                try: 
                    qa_record['short_answer'] = questions['answers'][0]['text']
                    num_with_ans += 1
                except:
                    qa_record['short_answer'] = ""
                    num_without_ans += 1
                    
                qa_records.append(qa_record)
        
        
    documents = [{'document_title':title, 'document_text': text}\
                         for title, text in documents.items()]
                
    print(f'Data contains {num_with_ans} question/answer pairs with a short answer, and {num_without_ans} without.'+
          f'\nThere are {len(documents)} unique wikipedia article paragraphs.')
                
    return qa_records, documents

def search_es(es_obj, index_name, question_text, n_results):
    '''
    Execute an Elasticsearch query on a specified index
    
    Args:
        es_obj (elasticsearch.client.Elasticsearch) - Elasticsearch client object
        index_name (str) - Name of index to query
        query (dict) - Query DSL
        n_results (int) - Number of results to return
        
    Returns
        res - Elasticsearch response object
    
    '''
    
    # construct query
    query = {
            'query': {
                'match': {
                    'document_text': question_text
                    }
                }
            }
    
    res = es_obj.search(index=index_name, body=query, size=n_results)
    
    return res

def searchcontext(question):
    res = search_es(es_obj=es, index_name=index_name, question_text=question, n_results=10)

    # get list of relevant context
    # titles = [(hit['_source']['document_title'], hit['_score']) for hit in res['hits']['hits']]
    # context = [hit['_source']['document_text'] for hit in res['hits']['hits']]

    # get the most relevent context
    context = res["hits"]["hits"][0]['_source']["document_text"]
    app.logger.info(context)

    return context

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            app.logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            app.logger.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            app.logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            app.logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text

def answerthequestion(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")

    answer_start_scores, answer_end_scores = model(**inputs)
    answer_start = torch.argmax(answer_start_scores)  # get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # get the most likely end of answer with the argmax of the score

    pred_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    final_answer = get_final_text(pred_answer, context, do_lower_case=True)

    return final_answer

# flask config
app = Flask(__name__, template_folder='templates', static_folder='assets')
app.config['SECRET_KEY'] = "tue khuyen"

@app.route('/')
def index():
    return render_template("index.html")

# flask routing
@app.route('/api/get_answer', methods=["POST"])
def get_answer():
    question = request.form["question"]

    try:
        context = searchcontext(question)
    except:
        return {
            "code" : -1,
            "answer" : "Xin lỗi tôi không thể trả lời câu hỏi này"
        }

    answer = answerthequestion(question, context)

    return {
        "code" : 0,
        "answer" : answer
    }

if __name__ == '__main__':
    # interacting with elasticsearch
    config = {'host':'localhost', 'port':9200}
    es = Elasticsearch([config])

    # create index
    index_config = {
    "settings": {
        "analysis": {
            "analyzer": {
                "stop_stem_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter":[
                        "lowercase",
                        "stop",
                        "snowball"
                    ]
                    
                }
            }
        }
    },
    "mappings": {
        "dynamic": "strict", 
        "properties": {
            "document_title": {"type": "text", "analyzer": "stop_stem_analyzer"},
            "document_text": {"type": "text", "analyzer": "stop_stem_analyzer"}
            }
        }
    }

    index_name = 'squad-standard-index'
    es.indices.create(index=index_name, body=index_config, ignore=400)

    # load corpus
    corpus_file = "../data/data.json"
    corpus = json.load(open(corpus_file, 'rb'))

    # parse question/answer record and get documents
    qa_records, documents = parse_qa_records(corpus['data'])

    # populate index for elasticsearch
    populate_index(es_obj=es, index_name=index_name, evidence_corpus=documents)

    # load the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained("../model")
    model = AutoModelForQuestionAnswering.from_pretrained("../model")

    app.run(debug=False)
