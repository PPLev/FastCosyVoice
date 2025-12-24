#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы методов inference_instruct2 в CosyVoice3

Метод inference_instruct2:
- Позволяет контролировать стиль генерации через текстовые инструкции
- Требует аудио-референс (prompt_wav) для клонирования голоса
- Формат instruct_text: "You are a helpful assistant. <инструкция><|endofprompt|>"

Тесты проверяют:
1. Инструкции на китайском языке
2. Инструкции на английском языке
3. Смешанные инструкции
"""

import sys
import os
sys.path.append('third_party/Matcha-TTS')

import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging

def test_instruct2_examples():
    """
    Тестирование различных инструкций с inference_instruct2
    """
    print("=" * 80)
    print("Инициализация модели CosyVoice3...")
    print("=" * 80)
    
    # Проверяем наличие модели
    model_dir = 'pretrained_models/Fun-CosyVoice3-0.5B'
    if not os.path.exists(model_dir):
        logging.error(f"Модель не найдена: {model_dir}", exc_info=True)
        return
    
    # Проверяем наличие референсного аудио
    prompt_wav = './refs/audio.wav'
    if not os.path.exists(prompt_wav):
        logging.error(f"Референсное аудио не найдено: {prompt_wav}", exc_info=True)
        return
    
    try:
        # Загружаем модель
        cosyvoice = AutoModel(model_dir=model_dir)
        print(f"✓ Модель загружена успешно")
        print(f"✓ Sample rate: {cosyvoice.sample_rate} Hz")
        print()
        
        # Создаём директорию для результатов
        output_dir = 'output/test_instruct'
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Результаты будут сохранены в: {output_dir}")
        print()
        
        # Тестовый текст на русском
        test_text_ru = "Привет, меня зовут Фаст Кози. Сегодня прекрасная погода и я очень рада вас видеть."
        
        # Более длинный русский текст
        test_text_ru_long = "Получив подарок на день рождения от друга издалека, я был приятно удивлён и глубоко тронут этим знаком внимания."
        
        # ============================================================
        # ТЕСТЫ С КИТАЙСКИМИ ИНСТРУКЦИЯМИ (русский текст)
        # ============================================================
        test_cases_chinese = [
            {
                'name': 'ru_cn_speed_fast',
                'instruction': 'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>',
                'description': 'Быстрая речь - китайская инструкция, русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_cn_speed_slow',
                'instruction': 'You are a helpful assistant. 请用尽可能慢地语速说一句话。<|endofprompt|>',
                'description': 'Медленная речь - китайская инструкция, русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_cn_emotion_happy',
                'instruction': 'You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>',
                'description': 'Радостная эмоция - китайская инструкция, русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_cn_emotion_sad',
                'instruction': 'You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>',
                'description': 'Грустная эмоция - китайская инструкция, русский текст',
                'text': test_text_ru
            },
        ]
        
        # ============================================================
        # ТЕСТЫ С АНГЛИЙСКИМИ ИНСТРУКЦИЯМИ (русский текст)
        # ============================================================
        test_cases_english = [
            {
                'name': 'ru_en_volume_loud',
                'instruction': 'You are a helpful assistant. Please say a sentence as loudly as possible.<|endofprompt|>',
                'description': 'Громко - английская инструкция, русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_en_volume_soft',
                'instruction': 'You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>',
                'description': 'Тихо - английская инструкция, русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_en_speed_fast',
                'instruction': 'You are a helpful assistant. Please speak as fast as possible.<|endofprompt|>',
                'description': 'Быстро - английская инструкция (кастомная), русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_en_speed_slow',
                'instruction': 'You are a helpful assistant. Please speak very slowly and clearly.<|endofprompt|>',
                'description': 'Медленно - английская инструкция (кастомная), русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_en_emotion_happy',
                'instruction': 'You are a helpful assistant. Please say this sentence in a very happy and excited tone.<|endofprompt|>',
                'description': 'Радостно - английская инструкция (кастомная), русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_en_emotion_sad',
                'instruction': 'You are a helpful assistant. Please say this sentence in a sad and melancholic tone.<|endofprompt|>',
                'description': 'Грустно - английская инструкция (кастомная), русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_en_emotion_angry',
                'instruction': 'You are a helpful assistant. Please say this sentence in an angry and frustrated tone.<|endofprompt|>',
                'description': 'Злобно - английская инструкция (кастомная), русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_en_whisper',
                'instruction': 'You are a helpful assistant. Please whisper this sentence.<|endofprompt|>',
                'description': 'Шёпот - английская инструкция (кастомная), русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_en_burr',
                'instruction': 'You are a helpful assistant. Please pronounce the letter R with a uvular trill, like a French R or a speech impediment where R sounds guttural.<|endofprompt|>',
                'description': 'Картавость - английская инструкция (кастомная), русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_en_lisp',
                'instruction': 'You are a helpful assistant. Please speak with a lisp, pronouncing S and Z sounds as TH.<|endofprompt|>',
                'description': 'Шепелявость - английская инструкция (кастомная), русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_en_no_r',
                'instruction': 'You are a helpful assistant. Please skip or omit the letter R completely when speaking.<|endofprompt|>',
                'description': 'Без буквы Р - английская инструкция (кастомная), русский текст',
                'text': test_text_ru
            },
        ]
        
        # ============================================================
        # ТЕСТЫ С РУССКИМИ ИНСТРУКЦИЯМИ (проверка)
        # ============================================================
        test_cases_russian_instruct = [
            {
                'name': 'ru_ru_speed_fast',
                'instruction': 'You are a helpful assistant. Пожалуйста, говорите как можно быстрее. <|endofprompt|>',
                'description': 'Быстро - русская инструкция, русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_ru_speed_slow',
                'instruction': 'You are a helpful assistant. Пожалуйста, говорите очень медленно и чётко. <|endofprompt|>',
                'description': 'Медленно - русская инструкция, русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_ru_emotion_happy',
                'instruction': 'You are a helpful assistant. Пожалуйста, скажите это очень радостным и весёлым голосом. <|endofprompt|>',
                'description': 'Радостно - русская инструкция, русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_ru_emotion_sad',
                'instruction': 'You are a helpful assistant. Пожалуйста, скажите это грустным голосом. <|endofprompt|>',
                'description': 'Грустно - русская инструкция, русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_ru_whisper',
                'instruction': 'You are a helpful assistant. Пожалуйста, прошепчите это предложение. <|endofprompt|>',
                'description': 'Шёпот - русская инструкция, русский текст',
                'text': test_text_ru
            },
            {
                'name': 'ru_ru_volume_soft',
                'instruction': 'You are a helpful assistant. Пожалуйста, скажите это очень тихим голосом. <|endofprompt|>',
                'description': 'Тихо - русская инструкция, русский текст',
                'text': test_text_ru
            },
        ]
        
        all_tests = [
            ("КИТАЙСКИЕ ИНСТРУКЦИИ + РУССКИЙ ТЕКСТ", test_cases_chinese),
            ("АНГЛИЙСКИЕ ИНСТРУКЦИИ + РУССКИЙ ТЕКСТ", test_cases_english),
            ("РУССКИЕ ИНСТРУКЦИИ + РУССКИЙ ТЕКСТ (эксперимент)", test_cases_russian_instruct),
        ]
        
        total_tests = sum(len(cases) for _, cases in all_tests)
        current_test = 0
        
        for section_name, test_cases in all_tests:
            print("=" * 80)
            print(f"СЕКЦИЯ: {section_name}")
            print("=" * 80)
            print()
            
            for test_case in test_cases:
                current_test += 1
                name = test_case['name']
                instruction = test_case['instruction']
                description = test_case['description']
                text = test_case['text']
                
                print(f"[{current_test}/{total_tests}] Тест: {description}")
                print(f"    Инструкция: {instruction}")
                print(f"    Текст: {text[:50]}..." if len(text) > 50 else f"    Текст: {text}")
                
                try:
                    # Генерируем аудио
                    for i, j in enumerate(cosyvoice.inference_instruct2(
                        tts_text=text,
                        instruct_text=instruction,
                        prompt_wav=prompt_wav,
                        stream=False
                    )):
                        output_path = f'{output_dir}/{name}_{i}.wav'
                        torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)
                        print(f"    ✓ Сохранено: {output_path}")
                    
                    print()
                    
                except Exception as e:
                    logging.error(f"Ошибка при генерации {name}: {e}", exc_info=True)
                    print()
                    continue
        
        print("=" * 80)
        print("✓ Все тесты завершены!")
        print(f"✓ Результаты сохранены в: {output_dir}")
        print("=" * 80)
        
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}", exc_info=True)
        raise


def print_supported_instructions():
    """
    Выводит список всех поддерживаемых инструкций
    """
    print("\n")
    print("=" * 80)
    print("ПОДДЕРЖИВАЕМЫЕ ИНСТРУКЦИИ для inference_instruct2")
    print("=" * 80)
    print()
    
    print("ОФИЦИАЛЬНЫЕ ИНСТРУКЦИИ НА КИТАЙСКОМ:")
    print("  Диалекты:")
    dialects = [
        "广东话", "东北话", "甘肃话", "贵州话", "河南话", "湖北话",
        "湖南话", "江西话", "闽南话", "宁夏话", "山西话", "陕西话",
        "山东话", "上海话", "四川话", "天津话", "云南话"
    ]
    for d in dialects:
        print(f"    - 请用{d}表达。")
    
    print("\n  Скорость:")
    print("    - 请用尽可能快地语速说一句话。")
    print("    - 请用尽可能慢地语速说一句话。")
    
    print("\n  Эмоции:")
    print("    - 请非常开心地说一句话。")
    print("    - 请非常伤心地说一句话。")
    print("    - 请非常生气地说一句话。")
    
    print("\n" + "=" * 80)
    print("ОФИЦИАЛЬНЫЕ ИНСТРУКЦИИ НА АНГЛИЙСКОМ:")
    print("  - Please say a sentence as loudly as possible.")
    print("  - Please say a sentence in a very soft voice.")
    
    print("\n" + "=" * 80)
    print("ТЕСТИРУЕМЫЕ КАСТОМНЫЕ ИНСТРУКЦИИ НА АНГЛИЙСКОМ:")
    print("  - Please speak as fast as possible.")
    print("  - Please speak very slowly and clearly.")
    print("  - Please say this sentence in a very happy and excited tone.")
    print("  - Please say this sentence in a sad and melancholic tone.")
    print("  - Please say this sentence in an angry and frustrated tone.")
    print("  - Please whisper this sentence.")
    
    print("\n" + "=" * 80)
    print("ФОРМАТ ИНСТРУКЦИИ:")
    print('  "You are a helpful assistant. <инструкция><|endofprompt|>"')
    print("=" * 80)


def main():
    """
    Главная функция
    """
    print("\n🎤 ТЕСТИРОВАНИЕ INFERENCE_INSTRUCT2 🎤\n")
    print("Цель: проверить работу инструкций на китайском и английском языках\n")
    
    # Показываем поддерживаемые инструкции
    print_supported_instructions()
    
    # Запускаем тесты
    test_instruct2_examples()
    
    print("\n✨ Готово! ✨\n")


if __name__ == '__main__':
    main()
