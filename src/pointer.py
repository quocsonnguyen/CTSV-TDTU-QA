import json

with open('data.json') as f:
    d = json.load(f)

for i, t in enumerate(d['data']):
    for j, p in enumerate(t['paragraphs']):
        if len(p['context'].split()) > 400:
            print(i+1, j+1)