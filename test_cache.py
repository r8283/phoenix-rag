"""
בדיקות למטמון התשובות ולשדה model בשאלות כפולות, בלי Gemini ובלי חיפוש.
שימוש:  python -m unittest test_cache -v
"""
import unittest
from unittest import mock

import rag

STANDALONE = {"key": rag.cache_key("מה זה ביטוח תכולה?"), "prev_key": None, "question": "מה זה ביטוח תכולה?",
              "status": "answered", "answer": "תשובה שמורה על תכולה", "sources": ["https://www.fnx.co.il/a/"],
              "followups": ["מה לא מכוסה?"], "model": "gemini-x"}
FOLLOWUP = {"key": rag.cache_key("ותוך כמה זמן מקבלים תשובה?"), "prev_key": rag.cache_key("יש לי נזילה בבית, איך אני מגישה תביעה?"),
            "question": "ותוך כמה זמן מקבלים תשובה?", "status": "answered", "answer": "תשובה שמורה: 30 יום",
            "sources": ["https://www.fnx.co.il/b/"], "followups": [], "model": "gemini-x"}
BUSY = {"key": rag.cache_key("שאלה שנשמרה בטעות"), "prev_key": None, "question": "שאלה שנשמרה בטעות",
        "status": "llm_unavailable", "answer": "עמוס", "sources": [], "followups": [], "model": None}
LIVE = {"status": "answered", "answer": "תשובה חיה", "sources": [], "model": "live"}


def hist(q):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": "..."}]


class CacheTest(unittest.TestCase):
    def setUp(self):
        self._old = rag._cache, rag.USE_CACHE
        rag._cache = {(e["key"], e["prev_key"]): e for e in (STANDALONE, FOLLOWUP, BUSY)}
        rag.USE_CACHE = True
        self.core = mock.patch.object(rag, "_answer_core", return_value=dict(LIVE)).start()
        self.addCleanup(mock.patch.stopall)

    def tearDown(self):
        rag._cache, rag.USE_CACHE = self._old

    def test_same_question_from_cache_without_llm(self):
        for q in ["מה זה ביטוח תכולה?", "מה זה  ביטוח תכולה", "מה זה ביטוח תכולה ?!", " מה זה ביטוח תכולה. "]:
            with self.subTest(q=q):
                r = rag.answer(q, [])
                self.assertTrue(r.get("cached"))
                self.assertEqual(r["answer"], "תשובה שמורה על תכולה")
                self.assertEqual(r["sources"], ["https://www.fnx.co.il/a/"])
        self.core.assert_not_called()

    def test_quotes_normalized(self):
        self.assertEqual(rag.cache_key('מה מכסה ביטוח נסיעות לחו"ל?'), rag.cache_key("מה מכסה ביטוח נסיעות לחו״ל"))
        self.assertEqual(rag.cache_key("מה מכסה ביטוח נסיעות לחו׳ל"), rag.cache_key("מה מכסה ביטוח נסיעות לחו'ל?"))

    def test_different_question_goes_to_llm(self):
        r = rag.answer("מה זה ביטוח מבנה?", [])
        self.assertFalse(r.get("cached"))
        self.assertEqual(r["answer"], "תשובה חיה")
        self.core.assert_called_once()

    def test_followup_with_same_previous_question(self):
        r = rag.answer("ותוך כמה זמן מקבלים תשובה?", hist("יש לי נזילה בבית איך אני מגישה תביעה"))
        self.assertTrue(r.get("cached"))
        self.core.assert_not_called()

    def test_followup_with_other_previous_question_goes_to_llm(self):
        r = rag.answer("ותוך כמה זמן מקבלים תשובה?", hist("איך מגישים תביעה בביטוח רכב?"))
        self.assertFalse(r.get("cached"))
        self.core.assert_called_once()

    def test_followup_without_context_goes_to_llm(self):
        r = rag.answer("ותוך כמה זמן מקבלים תשובה?", [])
        self.assertFalse(r.get("cached"))
        self.core.assert_called_once()

    def test_standalone_question_mid_conversation_from_cache(self):
        r = rag.answer("מה זה ביטוח תכולה?", hist("איך מגישים תביעה בביטוח רכב?"))
        self.assertTrue(r.get("cached"))
        self.core.assert_not_called()

    def test_bad_status_never_served(self):
        r = rag.answer("שאלה שנשמרה בטעות", [])
        self.assertFalse(r.get("cached"))
        self.core.assert_called_once()

    def test_cache_can_be_disabled(self):
        rag.USE_CACHE = False
        r = rag.answer("מה זה ביטוח תכולה?", [])
        self.assertFalse(r.get("cached"))
        self.core.assert_called_once()

    def test_cached_result_is_a_copy(self):
        rag.answer("מה זה ביטוח תכולה?", [])["sources"].append("x")
        self.assertEqual(rag.answer("מה זה ביטוח תכולה?", [])["sources"], ["https://www.fnx.co.il/a/"])


class MultiModelTest(unittest.TestCase):
    def test_double_question_lists_part_models(self):
        rag_results = iter([{"status": "answered", "answer": "א", "sources": [], "model": "m1"},
                            {"status": "answered", "answer": "ב", "sources": [], "model": "m2"}])
        with mock.patch.object(rag, "USE_CACHE", False), \
             mock.patch.object(rag, "_answer_core", side_effect=lambda *a, **k: next(rag_results)):
            r = rag.answer("מה זה ביטוח דירה ומה זה ביטוח רכב?", [])
        self.assertEqual(sorted(r["model"].split(", ")), ["m1", "m2"])


if __name__ == "__main__":
    unittest.main()
