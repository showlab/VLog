from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

class Evaluator:
    def __init__(self) -> None:
        self.tokenizer = PTBTokenizer()
        self.scorer_list = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]
        self.evaluation_report = {}

    def do_the_thing(self, golden_reference, candidate_reference):
        golden_reference = self.tokenizer.tokenize(golden_reference)
        candidate_reference = self.tokenizer.tokenize(candidate_reference)
        
        # From this point, some variables are named as in the original code
        # I have no idea why they name like these
        # The original code: https://github.com/salaniz/pycocoevalcap/blob/a24f74c408c918f1f4ec34e9514bc8a76ce41ffd/eval.py#L51-L63
        for scorer, method in self.scorer_list:
            score, scores = scorer.compute_score(golden_reference, candidate_reference)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.evaluation_report[m] = sc
            else:
                self.evaluation_report[method] = score

golden_reference = [
    "The quick brown fox jumps over the lazy dog.",
    "The brown fox quickly jumps over the lazy dog.",
    "A sly brown fox jumps over the lethargic dog.",
    "The speedy brown fox leaps over the sleepy hound.",
    "A fast, brown fox jumps over the lazy dog.",
]
golden_reference = {k: [{'caption': v}] for k, v in enumerate(golden_reference)}

candidate_reference = [
    "A fast brown fox leaps above the tired dog.",
    "A quick brown fox jumps over the sleepy dog.",
    "The fast brown fox jumps over the lazy dog.",
    "The brown fox jumps swiftly over the lazy dog.",
    "A speedy brown fox leaps over the drowsy dog.",
]
candidate_reference = {k: [{'caption': v}] for k, v in enumerate(candidate_reference)}

evaluator = Evaluator()

evaluator.do_the_thing(golden_reference, candidate_reference)

print(evaluator.evaluation_report)