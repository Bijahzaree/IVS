# автоматический тест для lab3.py
# запуск:
# python -m unittest test_lab3.py

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import unittest
from lab3 import BAYES_AVAILABLE, state, parse_command, execute_command, BayesianCommandLearner

class TestCommandParser(unittest.TestCase):
    def setUp(self):
        self.original_state = state.copy()
        self.test_learner = None
        if BAYES_AVAILABLE:
            self.test_learner = BayesianCommandLearner()
            for text, act in [
                ("установи температуру", "set"),
                ("покажи громкость", "get"),
                ("увеличь свет", "increase"),
                ("уменьши громкость", "decrease"),
            ]:
                self.test_learner.add_training_example(text, act)
            self.test_learner.train()

    def tearDown(self):
        global state
        state.clear()
        state.update(self.original_state)

    # -------------------- тесты парсинга (успешные) --------------------
    def test_parse_set_temperature(self):
        cmd = parse_command("установи температуру 22 градуса")
        self.assertEqual(cmd['action'], 'set')
        self.assertEqual(cmd['target'], 'temperature')
        self.assertEqual(cmd['value'], 22)

    def test_parse_increase_volume(self):
        cmd = parse_command("увеличь громкость на 10")
        self.assertEqual(cmd['action'], 'increase')
        self.assertEqual(cmd['target'], 'volume')
        self.assertEqual(cmd['value'], 10)

    def test_parse_turn_on_light(self):
        cmd = parse_command("включи свет")
        self.assertEqual(cmd['action'], 'set')
        self.assertEqual(cmd['target'], 'light')
        self.assertEqual(cmd['value'], True)

    def test_parse_get_temperature(self):
        cmd = parse_command("покажи температуру")
        self.assertEqual(cmd['action'], 'get')
        self.assertEqual(cmd['target'], 'temperature')
        self.assertIsNone(cmd['value'])

    def test_parse_string_value_ac(self):
        cmd = parse_command("установи кондиционер cool")
        self.assertEqual(cmd['action'], 'set')
        self.assertEqual(cmd['target'], 'ac')
        self.assertEqual(cmd['value'], 'cool')

    def test_parse_make_temperature(self):
        cmd = parse_command("сделай температуру 25")
        self.assertEqual(cmd['action'], 'set')
        self.assertEqual(cmd['target'], 'temperature')
        self.assertEqual(cmd['value'], 25)

    # -------------------- тесты ошибок (исправленные) --------------------
    def test_parse_incomplete_command(self):
        cmd = parse_command("сделай что-нибудь")
        self.assertIn('error', cmd)                     # должна быть ошибка

    def test_parse_unknown_action(self):
        cmd = parse_command("прыгай выше")
        self.assertIn('error', cmd)                     # должна быть ошибка

    def test_parse_no_target(self):
        cmd = parse_command("установи 100")
        self.assertIn('error', cmd)                     # должна быть ошибка

    # -------------------- тесты выполнения --------------------
    def test_execute_set_temperature(self):
        cmd = {"action": "set", "target": "temperature", "value": 25}
        execute_command(cmd)
        self.assertEqual(state['temperature'], 25)

    def test_execute_increase_volume(self):
        original = state['volume']
        cmd = {"action": "increase", "target": "volume", "value": 5}
        execute_command(cmd)
        self.assertEqual(state['volume'], original + 5)

    def test_execute_decrease_volume(self):
        original = state['volume']
        cmd = {"action": "decrease", "target": "volume", "value": 3}
        execute_command(cmd)
        self.assertEqual(state['volume'], original - 3)

    def test_execute_turn_on_light(self):
        cmd = {"action": "set", "target": "light", "value": True}
        execute_command(cmd)
        self.assertTrue(state['light'])

    # -------------------- тесты байесовской сети --------------------
    @unittest.skipIf(not BAYES_AVAILABLE, "Библиотеки pgmpy/pandas не установлены")
    def test_bayes_predict_action(self):
        pred = self.test_learner.predict_action("установи громкость 50")
        self.assertEqual(pred, "set")
        pred = self.test_learner.predict_action("покажи режим")
        self.assertEqual(pred, "get")
        pred = self.test_learner.predict_action("подними яркость")
        self.assertEqual(pred, "increase")
        pred = self.test_learner.predict_action("убавь звук")
        self.assertEqual(pred, "decrease")

if __name__ == "__main__":
    unittest.main()