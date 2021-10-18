from allennlp.predictors import Predictor
from src import *
from flask import Flask, request, jsonify
from nltk.tokenize import sent_tokenize

app = Flask(__name__)
pred = PropagandaPredictor.from_path("modelfull/", cuda_device=0)



@app.route("/get_spans", methods=["POST"])
def predict_article():
    input_json = request.json
    sentences = sent_tokenize(input_json["text"])
    sent_offsets = [0]
    for i in range(1, len(sentences)):
        sent_offsets.append(sent_offsets[i - 1] + len(sentences[i - 1]))
    jsons = [{"text": sent, "sent_offset": off} for sent, off in zip(sentences, sent_offsets)]
    r = []
    for j in jsons:
        r.append(pred.predict_json(j))
    result = {"sentences": [sr["formatted_text"] for sr in r]}
    result["spans"] = sum([sr["spans"] for sr in r], [])
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7777)

    