# Скрипт для локальной сборки EXE файлов
# Запуск: python build_exe.py

import os
import subprocess
import sys

def install_pyinstaller():
    """Установка PyInstaller"""
    print("📦 Встановлення PyInstaller...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    print("✅ PyInstaller встановлено!")

def build_exe():
    """Сборка EXE файлов"""
    print("🔨 Початок збірки EXE файлів...")
    
    # Создание папки dist если не существует
    if not os.path.exists("dist"):
        os.makedirs("dist")
    
    # Команды для сборки
    commands = [
        # Графическая версия (без консоли)
        [
            "pyinstaller",
            "--onefile",
            "--windowed",
            "--name", "AudioEditor_v1.0.4",
            "--icon", "icon.ico" if os.path.exists("icon.ico") else None,
            "audio_recorder_editor.py"
        ],
        # Консольная версия
        [
            "pyinstaller",
            "--onefile",
            "--name", "AudioEditor_Console_v1.0.4",
            "audio_editor_replit.py"
        ]
    ]
    
    for i, cmd in enumerate(commands):
        # Убираем None значения
        cmd = [x for x in cmd if x is not None]
        
        print(f"🔨 Збірка файлу {i+1}/2...")
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Файл {i+1} зібрано успішно!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Помилка при збірці файлу {i+1}: {e}")
            return False
    
    print("\n🎉 Всі EXE файли зібрано успішно!")
    print("📁 Файли знаходяться в папці 'dist':")
    
    # Показываем созданные файлы
    if os.path.exists("dist"):
        for file in os.listdir("dist"):
            if file.endswith(".exe"):
                file_path = os.path.join("dist", file)
                size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"   📄 {file} ({size:.1f} MB)")
    
    return True

def main():
    print("🎵 Аудіо Рекордер та Редактор - Збірка EXE v1.0.4")
    print("=" * 60)
    
    try:
        # Проверяем наличие PyInstaller
        try:
            import PyInstaller
            print("✅ PyInstaller вже встановлено")
        except ImportError:
            install_pyinstaller()
        
        # Собираем EXE
        if build_exe():
            print("\n🚀 Готово! Тепер ви можете:")
            print("   1. Запустити EXE файли без встановлення Python")
            print("   2. Поділитися ними з іншими")
            print("   3. Завантажити на GitHub як Release")
        else:
            print("\n❌ Збірка не вдалася. Перевірте помилки вище.")
            
    except Exception as e:
        print(f"❌ Критична помилка: {e}")

if __name__ == "__main__":
    main()
