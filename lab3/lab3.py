# Плотников Алексей Васильевич
# ЭВМбз-22-1
# Разработать командный язык прикладного пакета.
# Брать примеры с aot.ru
# Самообучение: Сети Байеса. Книга об обработке естественного языка на Python - NLTK book

# Словарь команд из примера: 
# {"action": "set", "target": "temperature", "value": 22}
#
# "action" – действие (например, "set", "get", "increase", "decrease").
#
# "target" – объект управления (например, "temperature", "volume", "light").
#
# "value" – числовое или строковое значение (например, 22, "on", "off").

"""
Лабораторная работа: Анализ фраз естественного языка и выполнение команд.
Разработка командного языка на основе логического вывода.
Студент: Плотников Алексей Васильевич
группа: ЭВМбз-22-1
"""

import re
import spacy
import subprocess
import sys
from collections import deque
from typing import Optional, Dict, Any

# Глобальное состояние устройств
state = {
    "temperature": 20,    # градусы Цельсия
    "volume": 50,         # процент громкости
    "light": False,       # выключен
    "ac": "off",          # кондиционер: off / cool / heat
    "mode": "auto"        # режим работы
}

# Словарь допустимых строковых значений для target'ов
STRING_VALUES = {
    "ac": {"cool", "heat", "off", "auto", "dry", "fan"},
    "mode": {"auto", "manual", "eco", "night", "day"}
}

# ----------------------------------------------------------------------
# Импорт байесовской сети (для обучения)
# ----------------------------------------------------------------------
_tried_install_bayes = False
BAYES_AVAILABLE = False
DiscreteBayesianNetwork = None
MaximumLikelihoodEstimator = None
pd = None

# Функция вывода сообщения об ошибке загрузки байесовской сети
def printLoadBayesError(error=None):
    if error:
        print(f"Ошибка: {error}")
    print("Предупреждение: библиотеки pgmpy/pandas не установлены. Байесовская сеть недоступна.")
    print("Установите вручную: pip install pgmpy pandas")
 
def load_bayes():
    global _tried_install_bayes
    global BAYES_AVAILABLE
    global BAYES_AVAILABLE
    global DiscreteBayesianNetwork
    global MaximumLikelihoodEstimator
    global pd
    try:
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.estimators import MaximumLikelihoodEstimator
        import pandas as pd
        BAYES_AVAILABLE = True
        print("Библиотеки pgmpy/pandas успешно загружены.")
    except ImportError as err:
        if _tried_install_bayes:
            # Вторая неудача – не пытаемся устанавливать снова
            printLoadBayesError(err)
        else:
            # Первая неудача – пытаемся установить
            _tried_install_bayes = True
            print(f"Библиотеки pgmpy/pandas не найдены. Пытаюсь установить: pip install pgmpy pandas")
            try:
                subprocess.run(
                [sys.executable, "-m", "pip", "install", "pgmpy", "pandas"],
                    check=True,
                    capture_output=False,
                    text=True
                )
                print("Установка выполнена. Повторная попытка импорта...")
                # Рекурсивный вызов – теперь _tried_install_bayes_ == True
                return load_bayes()
            except subprocess.CalledProcessError as err:
                printLoadBayesError(err)

# непосредственно сам импорт байесовской сети                
load_bayes()

# ----------------------------------------------------------------------
# 1. Загрузка языковой модели spaCy (русский)
# ----------------------------------------------------------------------

_tried_install_model_ = False

# Функция вывода сообщения об ошибке загрузки языковой модели
def printLoadModelError(error=None):
    if error:
        print(f"Ошибка: {error}")
    print("Модель 'ru_core_news_sm' не найдена. Установите её командой:")
    print("python -m spacy download ru_core_news_sm")
    sys.exit(1)
 
# ----------------------------------------------------------------------
# Импорт языковой модели
# ---------------------------------------------------------------------- 
def load_spacy_model(model_name):
    global _tried_install_model_
    try:
        nlp = spacy.load(model_name)
        print(f"Модель '{model_name}' успешно загружена.")
        print()
        return nlp
    except OSError as err:
        if _tried_install_model_:
            # Вторая неудача – не пытаемся устанавливать снова
            printLoadModelError(err)
        else:
            # Первая неудача – пытаемся установить
            _tried_install_model_ = True
            print(f"Модель '{model_name}' не найдена. Устанавливаю...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", model_name],
                    check=True
                )
                print(f"Модель '{model_name}' установлена. Повторная попытка загрузки...")
                # Рекурсивный вызов – теперь _tried_install_model_ == True
                return load_spacy_model(model_name)
            except subprocess.CalledProcessError as err:
                printLoadModelError(err)

# загрузка самой языковой модели
nlp = load_spacy_model("ru_core_news_sm")

# ----------------------------------------------------------------------
# Байесовская сеть для самообучения
# ----------------------------------------------------------------------
class BayesianCommandLearner:
    def __init__(self):
        if not BAYES_AVAILABLE:
            self.model = None
            self.data = None
            return
        # Структура сети: признаки влияют на действие
        self.model = DiscreteBayesianNetwork([('has_set', 'action'),
                                      ('has_get', 'action'),
                                      ('has_inc', 'action'),
                                      ('has_dec', 'action')])
        self.data = pd.DataFrame(columns=['has_set', 'has_get', 'has_inc', 'has_dec', 'action'])
        self.training_examples = deque(maxlen=100)  # храним последние 100 примеров

    def _extract_features(self, text: str) -> dict:
        # Извлечение бинарных признаков на основе русских ключевых слов
        text_low = text.lower()
        # Признак "установить/изменить"
        has_set = any(word in text_low for word in ['установи', 'поставь', 'измени', 'сделай', 'выставь'])
        # Признак "получить/показать"
        has_get = any(word in text_low for word in ['покажи', 'получи', 'выведи', 'скажи'])
        # Признак "увеличить"
        has_inc = any(word in text_low for word in ['увеличь', 'подними', 'прибавь', 'повысь'])
        # Признак "уменьшить"
        has_dec = any(word in text_low for word in ['уменьши', 'опусти', 'убавь', 'понизь'])
        return {
            'has_set': 1 if has_set else 0,
            'has_get': 1 if has_get else 0,
            'has_inc': 1 if has_inc else 0,
            'has_dec': 1 if has_dec else 0,
        }

    def add_training_example(self, text: str, true_action: str):
        # Добавление примера в обучающую выборку
        if not BAYES_AVAILABLE:
            return
        # Устранение "избыточности" команд turn_on/turn_off -> set 
        if true_action in ["turn_on", "turn_off"]:
            true_action = "set"
        feats = self._extract_features(text)
        feats['action'] = true_action
        new_row = pd.DataFrame([feats])
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        self.training_examples.append((text, true_action))

    def train(self):
        # Обучение байесовской сети на накопленных данных
        if not BAYES_AVAILABLE or self.data.empty:
            return
        try:
            self.model.fit(self.data, estimator=MaximumLikelihoodEstimator)
            print("Байесовская сеть обучена на", len(self.data), "примерах.")
        except Exception as err:
            print(f"Ошибка обучения сети: {err}")

    def predict_action(self, text: str) -> str:
        # Предсказание действия для нового текста
        if not BAYES_AVAILABLE or self.data.empty:
            return None
        feats = self._extract_features(text)
        input_df = pd.DataFrame([feats])
        # Убедимся, что порядок колонок совпадает с обучением
        input_df = input_df[['has_set', 'has_get', 'has_inc', 'has_dec']]
        try:
            pred = self.model.predict(input_df)
            action = pred['action'].iloc[0]
            # Устранение "избыточности" команд turn_on/turn_off -> set 
            if action in ["turn_on", "turn_off"]:
                action = "set"
            return action
        except Exception:
            return None

# ----------------------------------------------------------------------
# 2. Функция разбора команды из фразы естественного языка
# ----------------------------------------------------------------------
def parse_command(phrase: str) -> dict:
    # Разбор фразы
    doc = nlp(phrase.lower().strip())
    
    action = _find_action(doc) # попытка найти действие
    target = _find_target(doc) # попытка найти цель (назначение)
    value = _find_value(phrase, doc) # попытка найти числовое значение
    if value is None and target is not None:
        value = _find_string_value(doc, target) # попытка найти строкове значение если не получлось с числами
        
    # Иногда хочется знать что происходит для отладки
    # print(f"DEBUG: action: {action} target: {target} value: {value}")
    
    # Обработка команд включения/выключения и get как команд с особым подходом
    match action:
        case "turn_on" | "turn_off":
            if value is None:
                value = True if action == "turn_on" else False
                if target == "ac":
                    value = "auto" if action == "turn_on" else "off"
            action = "set"
        case "get":
            value = None
        case _:
            # Если действие не найдено – пробуем байесовскую сеть
            if action is None and BAYES_AVAILABLE and 'learner' in globals() and learner.data is not None:
                predicted = learner.predict_action(phrase)
                if predicted:
                    action = predicted
                    print(f"Байесовская сеть предсказала действие: {action}")
    
    # Проверка успешности разбора
    if action and target:
        return {"action": action, "target": target, "value": value}
    else:
        return {
            "error": "Не удалось полностью разобрать команду",
            "parsed": {"action": action, "target": target, "value": value}
        }


def _find_action(doc) -> Optional[str]:
    # Поиск действия
    action_map = {
        "установи": "set", "поставь": "set", "измени": "set", "сделай": "set",
        "увеличь": "increase", "подними": "increase", "прибавь": "increase",
        "уменьши": "decrease", "опусти": "decrease", "убавь": "decrease",
        "включи": "turn_on", "выключи": "turn_off",
        "покажи": "get", "получи": "get", "выведи": "get"
    }
    
    # Сначала ищем по точному тексту, потом по лемме
    for token in doc:
        # Иногда хочется знать что происходит для отладки. Иногда помогает
        # print(f"DEBUG: TOKEN: {token.text}, POS: {token.pos_}")
        if token.text in action_map:
            return action_map[token.text]
        if token.lemma_ in action_map:
            return action_map[token.lemma_]
    return None


def _find_target(doc) -> Optional[str]:
    # Поиск цели (назначения)
    target_map = {
        "температура": "temperature", "температуру": "temperature",
        "громкость": "volume",
        "свет": "light", "освещение": "light",
        "кондиционер": "ac", "климат": "ac",
        "режим": "mode", "профиль": "mode"
    }
    stop_targets = {"градус", "градуса", "процент", "процента", "раз", "на", "единиц"}
    
    for token in doc:
        if token.pos_ != "NOUN":
            continue
        # Проверяем по лемме или тексту, исключая стоп-слова
        if token.lemma_ in target_map and token.lemma_ not in stop_targets:
            return target_map[token.lemma_]
        if token.text in target_map and token.text not in stop_targets:
            return target_map[token.text]
    return None


def _find_value(phrase: str, doc) -> Optional[Any]:
    # Посик числа
    # Сначала пробуем через токены spaCy
    for token in doc:
        if token.like_num:
            try:
                return float(token.text) if '.' in token.text else int(token.text)
            except ValueError:
                continue # бывает что эта ошибка и не ошибка вовсе...
    # Если не нашли - через регулярное выражение
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', phrase)
    if numbers:
        raw = numbers[0]
        return float(raw) if '.' in raw else int(raw)
    return None

def _find_string_value(doc, target: str) -> Optional[str]:
    # Поиск строки 
    # Проверка на допустимость строковых данных по STRING_VALUES
    if target not in STRING_VALUES:
        return None
    allowed = STRING_VALUES[target]
    for token in doc:
        # Ищем лемму или текст токена (приводим к нижнему регистру)
        if token.lemma_.lower() in allowed:
            return token.lemma_.lower()
        if token.text.lower() in allowed:
            return token.text.lower()
    return None
  
# ----------------------------------------------------------------------
# 3. Эмуляция состояния системы и выполнение команды
# ----------------------------------------------------------------------


def execute_command(command: dict) -> None:
    # Выполннеие команды
    if "error" in command:
        print(f"Ошибка разбора: {command['error']}")
        print(f"   Частичный разбор: {command.get('parsed', {})}")
        return

    action = command["action"]
    target = command["target"]
    value = command["value"]

    # Проверка существования целевого параметра
    if target not in state:
        print(f"Неизвестный параметр '{target}'. Доступные: {list(state.keys())}")
        return

    match action:
        case "set":
            state[target] = value
            print(f"Установлен {target} = {value}")

        case "increase":
            _change_numeric_state(state, target, value, delta=+1, operation_name="увеличен")

        case "decrease":
            _change_numeric_state(state, target, value, delta=-1, operation_name="уменьшен")

        case "get":
            print(f"Текущее значение {target}: {state[target]}")

        case _:
            print(f"Неизвестное действие '{action}'")


def _change_numeric_state(state: dict, target: str, value, delta: int, operation_name: str) -> None:
    # Операции изменения числовых параметров
    current = state[target]
    if isinstance(current, (int, float)):
        change = value if value is not None else 1
        state[target] = current + delta * change
        print(f"{target} {operation_name} до {state[target]}")
    else:
        print(f"❌ Невозможно {operation_name} '{target}', так как его тип {type(current)}")

# ----------------------------------------------------------------------
# Инициализация байесовской сети и обучение на начальных примерах
# ----------------------------------------------------------------------
learner = BayesianCommandLearner() if BAYES_AVAILABLE else None

# Начальное обучение на типичных командах (если сеть доступна)
if BAYES_AVAILABLE and learner:
    initial_examples = [
        ("установи температуру 22", "set"),
        ("покажи громкость", "get"),
        ("увеличь свет", "increase"),
        ("уменьши громкость", "decrease"),
        ("включи кондиционер", "set"),
        ("выключи свет", "set"),
        ("включи свет", "set"),
    ]
    for text, act in initial_examples:
        learner.add_training_example(text, act)
    learner.train()
    

# ----------------------------------------------------------------------
# 4. Демонстрация работы (тестовые примеры)
# ----------------------------------------------------------------------

# Код, который должен выполняться только при прямом запуске файла,
# но не при импорте этого файла как модуля в другую программу.
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Режим 1: Интерактивный ввод с клавиатуры (аргумент -keyb)
    # ------------------------------------------------------------------
    if "-keyb" in sys.argv:
        print("=" * 60)
        print("Интерактивный режим. Вводите команды (на русском).")
        print("Для выхода введите: quit, exit, выход, закончить")
        print("=" * 60)
        while True:
            user_input = input("\n> ").strip()
            if not user_input:
                continue
            # Проверка команд выхода
            if user_input.lower() in ("quit", "exit", "выход", "закончить"):
                print("Завершение работы.")
                break
            # Разбор и выполнение команды
            command = parse_command(user_input)
            print(f"Разобранная команда: {command}")
            execute_command(command)
            # Обучение на успешной команде
            if "error" not in command and BAYES_AVAILABLE and learner:
                learner.add_training_example(user_input, command['action'])
                learner.train()
            # Вывод текущего состояния (опционально)
            print(f"Текущее состояние: {state}")
    # ------------------------------------------------------------------
    # Режим 2: Передача одной команды через аргументы командной строки
    # ------------------------------------------------------------------
    # Если переданы аргументы командной строки
    if len(sys.argv) > 1:
        # Объединяем все аргументы в одну строку (поддерживает фразы без кавычек)
        user_phrase = " ".join(sys.argv[1:])
        print(f"Команда из командной строки: \"{user_phrase}\"")
        command = parse_command(user_phrase)
        print(f"Разобранная команда: {command}")
        execute_command(command)
        # После выполнения добавляем пример в обучение (если команда успешна)
        if "error" not in command and BAYES_AVAILABLE and learner:
            learner.add_training_example(user_phrase, command['action'])
            learner.train()   # переобучаем с новым примером
    # ------------------------------------------------------------------
    # Режим 3: Демонстрация тестовых примеров (по умолчанию)
    # ------------------------------------------------------------------
    else:
        # Нет аргументов – запускаем тестовые примеры
        print("=" * 60)
        print("Командный анализатор естественного языка (русский)")
        print("=" * 60)
        
        test_phrases = [
            "установи температуру 22 градуса",
            "увеличь громкость на 10",
            "включи свет",
            "покажи температуру",
            "уменьши громкость на 5",
            "выключи свет",
            "получи режим",
            "установи кондиционер cool",
            "сделай температуру 25",
        ]
        
        for phrase in test_phrases:
            print(f"\nФраза: \"{phrase}\"")
            cmd = parse_command(phrase)
            print(f"Разобранная команда: {cmd}")
            execute_command(cmd)
            print(f"Текущее состояние: {state}")
        
        # Пример неполной команды
        print("\n" + "-" * 60)
        bad_phrase = "сделай что-нибудь"
        print(f"Фраза: \"{bad_phrase}\"")
        cmd = parse_command(bad_phrase)
        print(f"Разобранная команда: {cmd}")
        execute_command(cmd)
        
    # Финальное обучение и вывод статистики
    if BAYES_AVAILABLE and learner:
        print("\nСтатистика байесовской сети:")
        print(f"   Всего примеров: {len(learner.data)}")
        if not learner.data.empty:
            print(learner.data['action'].value_counts())
