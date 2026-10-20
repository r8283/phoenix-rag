"""
בדיקות ל-_is_personal_question ולהפניה המיידית לשירות הלקוחות, בלי Gemini ובלי חיפוש.
שימוש:  python -m unittest test_personal -v
"""
import unittest
from unittest import mock

import rag

PERSONAL = [
    "מה הפוליסה האישית שלי?",
    "מה סטטוס התביעה שלי?",
    "כמה שילמתי השנה על הביטוח שלי?",
    "האם הפוליסה שלי מכסה את הנזק?",
]
GENERAL = [
    "יש לי נזילה בבית, איך אני מגישה תביעה?",
    "אפשר לקחת הלוואה מהחיסכון שלי?",
    "מה זה קרן פנסיה?",
    "איך מגישים תביעה בביטוח רכב?",
    "מה הכיסוי בביטוח תכולה?",
]


class PersonalQuestionTest(unittest.TestCase):
    def test_personal(self):
        for q in PERSONAL:
            with self.subTest(q=q):
                self.assertTrue(rag._is_personal_question(q))

    def test_general(self):
        for q in GENERAL:
            with self.subTest(q=q):
                self.assertFalse(rag._is_personal_question(q))

    def test_answer_skips_search_and_llm(self):
        with mock.patch.object(rag, "_answer_core") as core:
            for q in PERSONAL:
                with self.subTest(q=q):
                    r = rag.answer(q, [])
                    self.assertEqual(r["status"], "personal_only")
                    self.assertEqual(r["answer"], rag.PERSONAL_ONLY_TEXT)
                    self.assertEqual(r["sources"], [])
            core.assert_not_called()


if __name__ == "__main__":
    unittest.main()
