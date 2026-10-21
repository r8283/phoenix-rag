"""
בדיקות ל-_clean_followups, בלי Gemini ובלי מודל ההטמעה (האחזור מוחלף בקטע קבוע).
שימוש:  python -m unittest test_followups -v
"""
import json
import unittest
from unittest import mock

import rag

GOOD = "מה כוללת הלוואה כנגד חיסכון?"
LOW = "איך מחשבים שטח של מעגל?"
# הטקסט של "הקטעים שנשלפו" בכל הבדיקות
PASSAGE = (
    "הלוואה כנגד חיסכון כוללת ריבית לפי מסלול. לא צריך ערבים או ביטחונות. "
    "ישנם שני סוגי קרנות פנסיה: קרן פנסיה מקיפה וקרן פנסיה כללית. הלוואה מספר"
)


def fake_retrieve(query, k=17):
    score = 0.70 if query == LOW else 0.90
    return [{"score": score, "url": "u", "title": "t", "text": PASSAGE}]


@mock.patch.object(rag, "retrieve", side_effect=fake_retrieve)
class CleanFollowupsTest(unittest.TestCase):
    Q = "מה זה קרן פנסיה?"

    def test_not_a_list(self, _):
        self.assertEqual(rag._clean_followups(None, self.Q), [])
        self.assertEqual(rag._clean_followups(GOOD, self.Q), [])
        self.assertEqual(rag._clean_followups({"q": GOOD}, self.Q), [])

    def test_non_string_items_skipped(self, _):
        self.assertEqual(rag._clean_followups([3, None, GOOD], self.Q), [GOOD])

    def test_requires_question_mark(self, _):
        self.assertEqual(rag._clean_followups(["מה כוללת הלוואה כנגד חיסכון"], self.Q), [])

    def test_length_limits(self, _):
        self.assertEqual(rag._clean_followups(["מה זה?", "מה " * 30 + "?"], self.Q), [])

    def test_foreign_nikud_garbled(self, _):
        bad = ["מה זה пенсия בקרן?", "מה זה קֶרֶן פנסיה?", "מה זה קרן  פנסיה?"]
        self.assertEqual(rag._clean_followups(bad, self.Q), [])

    def test_same_as_question(self, _):
        self.assertEqual(rag._clean_followups(["מה זה  קרן פנסיה ??"], self.Q), [])

    def test_duplicates(self, _):
        self.assertEqual(rag._clean_followups([GOOD, GOOD, " " + GOOD], self.Q), [GOOD])

    def test_low_retrieval_score(self, _):
        self.assertEqual(rag._clean_followups([LOW, GOOD], self.Q), [GOOD])

    def test_term_missing_from_passages_rejected(self, _):
        # "סיעוד" ו"מחלות" לא מופיעים בקטעים: רק 1 מ-3 מילות תוכן מכוסה
        self.assertEqual(rag._clean_followups(["האם ביטוח סיעוד מכסה מחלות?"], self.Q), [])

    def test_all_terms_in_passages_pass(self, _):
        q = "מה ההבדל בין קרן פנסיה מקיפה לכללית?"
        self.assertEqual(rag._clean_followups([q], self.Q), [q])

    def test_max_three(self, _):
        items = [f"מה כוללת הלוואה מספר {i}?" for i in range(5)]
        self.assertEqual(rag._clean_followups(items, self.Q), items[:3])


class RefusalFollowupsTest(unittest.TestCase):
    """סירוב שהגיע מהמודל מקבל את ההצעות שלו (אחרי סינון). סירוב בלי פנייה למודל לא מקבל."""
    Q = "מה ההבדל בין קופת גמל לקרן השתלמות?"
    GOOD_FU = "מה ההבדל בין קופת גמל להשקעה לקרן השתלמות?"
    BAD_FU = "האם ביטוח סיעוד מכסה מחלות כרוניות?"
    TEXT = "מה ההבדל בין קופת גמל להשקעה לקרן השתלמות? קרן השתלמות נזילה לאחר 6 שנים, וקופת גמל להשקעה נזילה בכל עת."

    def setUp(self):
        self.score = 0.90
        p = [mock.patch.object(rag, "USE_CACHE", False),
             mock.patch.object(rag, "rewrite_question", side_effect=lambda q, h: q),
             mock.patch.object(rag, "retrieve", side_effect=lambda q, k=rag.TOP_K: [
                 {"score": self.score, "url": "https://www.fnx.co.il/gemel/gemel-information/gemel-vs-hishtalmot/",
                  "title": "t", "text": self.TEXT}])]
        for x in p:
            x.start()
        self.addCleanup(mock.patch.stopall)

    def model_says(self, followups):
        raw = json.dumps({"status": "not_found", "answer": "...", "used": [1], "personal": False, "followups": followups},
                         ensure_ascii=False)
        return mock.patch.object(rag, "call_llm", return_value=(raw, "fake-model"))

    def test_model_refusal_keeps_relevant_followup(self):
        with self.model_says([self.GOOD_FU]):
            r = rag.answer(self.Q, [])
        self.assertEqual(r["status"], "not_found")
        self.assertEqual(r["answer"], rag.NOT_FOUND_TEXT)
        self.assertEqual(r["sources"], [])
        self.assertEqual(r["followups"], [self.GOOD_FU])

    def test_model_refusal_filters_irrelevant_followup(self):
        with self.model_says([self.BAD_FU, self.GOOD_FU]):
            r = rag.answer(self.Q, [])
        self.assertEqual(r["status"], "not_found")
        self.assertEqual(r["followups"], [self.GOOD_FU])

    def test_low_score_refusal_has_no_followups_and_no_llm(self):
        self.score = 0.70
        with self.model_says([self.GOOD_FU]) as llm:
            r = rag.answer(self.Q, [])
        self.assertEqual(r["status"], "not_found")
        self.assertEqual(r.get("followups") or [], [])
        llm.assert_not_called()

    def test_personal_and_meta_have_no_followups(self):
        with self.model_says([self.GOOD_FU]) as llm:
            for q in ["מה סטטוס התביעה שלי?", "שלום איך קוראים לך"]:
                with self.subTest(q=q):
                    self.assertEqual(rag.answer(q, []).get("followups") or [], [])
        llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
