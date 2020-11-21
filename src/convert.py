import json

with open("data/squad_vi_v2.json", encoding="utf-8-sig") as f:
    d = json.load(f)

with open("new.json" , "w", encoding="utf-8") as out:
    json.dump(d, out, indent=4 ,ensure_ascii=False)