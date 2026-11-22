import unittest
from unittest.mock import patch
from pathlib import Path
from k3_llama import configuration, fixed_prompt, check_round


class BenchmarkTests(unittest.TestCase):
    def test_profiles_change_device_separately_from_batch(self):
        def settings(name):
            return configuration(Path("/runtime/bin/llama-server"), Path("/model.gguf"), name, 8000, [])
        old, oldenv = settings("legacy64")
        with patch.dict("os.environ", {"GGML_SPACEMIT_WORKERS": "2"}):
            new, newenv = settings("standalone64")
        self.assertEqual(newenv["GGML_SPACEMIT_WORKERS"], "8")
        wide, _ = settings("standalone128")
        self.assertEqual(old[old.index("-ub") + 1], new[new.index("-ub") + 1])
        self.assertEqual(wide[wide.index("-ub") + 1], "128")
        self.assertEqual(new[new.index("--device") + 1], "SPACEMIT0")
        self.assertEqual(oldenv["GGML_SPACEMIT_BACKEND"], "0")
        self.assertEqual(newenv["GGML_SPACEMIT_BACKEND"], "1")

    def test_prompt_length_and_suffix(self):
        ids = fixed_prompt([7, 8], list(range(200)), [11, 12])
        self.assertEqual(len(ids), 128)
        self.assertEqual(len(fixed_prompt([7], list(range(600)), [3], 512)), 512)
        self.assertEqual(ids[:2], [7, 8])
        self.assertEqual(ids[-2:], [11, 12])
        with self.assertRaises(ValueError):
            fixed_prompt([1], [2], [3])

    def test_cached_or_short_runs_do_not_enter_medians(self):
        good = dict(prompt_n=128, predicted_n=64, cache_n=0)
        check_round(good)
        with self.assertRaises(ValueError):
            check_round({"prompt_n": 128, "predicted_n": 64})
        check_round(dict(good, prompt_n=512), 512)
        with self.assertRaises(ValueError):
            check_round(good, 512)
        for invalid in (dict(cache_n=1), dict(predicted_n=63), dict(prompt_n=127)):
            with self.assertRaises(ValueError):
                check_round(dict(good, **invalid))


if __name__ == "__main__":
    unittest.main()
