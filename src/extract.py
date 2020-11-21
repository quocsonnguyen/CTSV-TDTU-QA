import json

with open("./data/vi_squad_v1.json", encoding="utf-8") as f:
    data = json.load(f)

print(len(data["data"][0]["paragraphs"]))

count = 7500
para = []

for i in range(50):
    if count == 7050:
        break

    para.append(data["data"][0]["paragraphs"][count])

    count += 1

# for i in range(len(data["data"][0]["paragraphs"])):
#     if count == 7050:
#         break

#     para.append(data["data"][0]["paragraphs"][i+7000])

#     count += 1

newdata = {
    "version": "v1.1",
    "data": [
        {
            "title": "translated",
            "paragraphs": para
        }
    ]
}

with open("./data/test.json", "w", encoding="utf-8") as f:
    json.dump(newdata, f, indent=4, ensure_ascii=False)