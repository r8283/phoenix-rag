"""
בדיקות ל-_meta_reply (שאלות על הבוט עצמו, ברכות ותודה), בלי Gemini ובלי חיפוש.
שימוש:  python -m unittest test_meta -v
"""
import unittest

import rag

META = [
    "איך קוראים לך?",
    "שלום איך קוראים לך",
    "היי מה שמך",
    "שלום, מי אתה?",
    "במה אתה עוזר?",
    "שלום",
    "מה שלומך",
    "תודה רבה",
]
NOT_META = [
    "איך קוראים לראש הממשלה?",
    "מה זה קרן פנסיה?",
    "איך מגישים תביעה?",
    "שלום, איך מגישים תביעה בביטוח רכב?",
    "מה אתה ממליץ, מקיף או צד ג?",
    "?",
    "",
]


class MetaReplyTest(unittest.TestCase):
    def test_meta(self):
        for q in META:
            with self.subTest(q=q):
                r = rag._meta_reply(q)
                self.assertIsNotNone(r)
                self.assertEqual(r["status"], "meta")
                self.assertEqual(r["sources"], [])

    def test_not_meta(self):
        for q in NOT_META:
            with self.subTest(q=q):
                self.assertIsNone(rag._meta_reply(q))

    def test_replies(self):
        self.assertEqual(rag._meta_reply("שלום איך קוראים לך")["answer"], rag._META_NAME)
        self.assertTrue(rag._meta_reply("שלום")["answer"].endswith("במה אפשר לעזור?"))
        self.assertEqual(
            rag._meta_reply("תודה רבה")["answer"],
            "בשמחה. אם יש עוד שאלה על ביטוח, פנסיה, חיסכון או הלוואות, אפשר לשאול.",
        )


if __name__ == "__main__":
    unittest.main()
