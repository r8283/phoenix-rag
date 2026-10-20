"""
בדיקות ל-_clean_followups, בלי Gemini ובלי מודל ההטמעה (האחזור מוחלף בציון קבוע).
שימוש:  python -m unittest test_followups -v
"""
import unittest
from unittest import mock

import rag

GOOD = "מה כוללת הלוואה כנגד חיסכון?"
LOW = "איך מחשבים שטח של מעגל?"


def fake_retrieve(query, k=17):
    score = 0.70 if query == LOW else 0.90
    return [{"score": score, "url": "u", "title": "t", "text": "x"}]


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

    def test_max_three(self, _):
        items = [f"מה כוללת הלוואה מספר {i}?" for i in range(5)]
        self.assertEqual(rag._clean_followups(items, self.Q), items[:3])


if __name__ == "__main__":
    unittest.main()
