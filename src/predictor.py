import numpy as np
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token

from src.utils import generate_spans
from src.process_data_si import insert_spans



@Predictor.register("prop_pred")
class PropagandaPredictor(Predictor):

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokenizer = self._dataset_reader.tokenizer
        tokens = tokenizer.tokenize(json_dict["text"])
        tokens = [Token("[CLS]", text_id=101, idx=0)] + tokens
        tokens.append(Token("[SEP]", text_id=10, idx=tokens[-1].idx))
        instance = self._dataset_reader.text_to_instance(tokens)
        instance.fields["metadata"].metadata["article_text"] = json_dict["text"]
        instance.fields["metadata"].metadata["sent_offset"] = json_dict.get("sent_offset", 0)
        return instance
        
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        res = self.predict_instance(instance)
        tags = np.argmax(res["tag_logits"], axis=1).tolist()
        tokens = instance.fields["metadata"].metadata["tokens"]
        spans = generate_spans(tokens, tags, return_spans=True)
        spans_ret = generate_spans(tokens, tags, return_spans=True, sentence_offset=inputs.get("sent_offset"))
        res["spans"] = spans_ret
        res["formatted_text"] = insert_spans(inputs["text"], spans)[0].split("\t")[0]
        return res