from flask import Flask, request, render_template
import numpy as np
import collections
import torch
import json

from rank_bm25 import BM25Okapi
from transformers import BasicTokenizer ,AutoTokenizer, AutoModelForQuestionAnswering

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

def search_context(question):
    score = np.sum(bm25.get_scores(question.split()))
    if score <= 5:
        return -1
    return bm25.get_top_n(question.split(), CONTEXTS, n=1)[0]

def answer_the_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")

    output = model(**inputs)
    answer_start = torch.argmax(output.start_logits)  # get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(output.end_logits) + 1  # get the most likely end of answer with the argmax of the score

    pred_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    final_answer = get_final_text(pred_answer, context, do_lower_case=True)

    return final_answer

# flask config
app = Flask(__name__, template_folder='templates', static_folder='assets')
app.config['SECRET_KEY'] = "monn monn"

@app.route('/')
def index():
    return render_template("index.html")

# flask routing
@app.route('/api/get_answer', methods=["POST"])
def get_answer():
    question = request.form["question"]

    try:
        context = search_context(question)
        if context == -1:
            return {
                "code" : -1,
                "answer" : "Xin lỗi tôi không thể trả lời câu hỏi này"
            }  
    except:
        return {
            "code" : -1,
            "answer" : "Xin lỗi tôi không thể trả lời câu hỏi này"
        }

    answer = answer_the_question(question, context)

    return {
        "code" : 0,
        "answer" : answer
    }

if __name__ == '__main__':

    # load corpus
    corpus_file = "./data/data.json"
    corpus = json.load(open(corpus_file, 'rb'))

    CONTEXTS = []
    for doc in corpus['data']:
        for p in doc['paragraphs']:
            CONTEXTS.append(p['context'])

    tokenized_contexts = [context.split() for context in CONTEXTS]

    # Load BM25 for searching context
    bm25 = BM25Okapi(tokenized_contexts)

    # load the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained("./model")
    model = AutoModelForQuestionAnswering.from_pretrained("./model")

    app.run(debug=False)
