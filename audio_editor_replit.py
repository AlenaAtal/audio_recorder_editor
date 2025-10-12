# Аудіо Рекордер та Редактор v1.0.4 для Replit
# Запуск: python audio_editor_replit.py

import os
import sys
import numpy as np
import tempfile
from pydub import AudioSegment, effects
from pydub.silence import detect_nonsilent
import noisereduce as nr
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Для Replit

# Функція для обрізки тиші
def trim_silence(audio, silence_thresh=-40, min_silence_len=100):
    nonsilent = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not nonsilent:
        return audio
    start_trim = nonsilent[0][0]
    end_trim = nonsilent[-1][1]
    return audio[start_trim:end_trim]

# Функція для створення еквалайзера
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_equalizer(samples, fs, gains):
    # 5 смуг: 60, 250, 1000, 4000, 16000 Гц
    bands = [(20, 120), (120, 500), (500, 2000), (2000, 8000), (8000, 20000)]
    out = np.zeros_like(samples)
    for i, (low, high) in enumerate(bands):
        b, a = butter_bandpass(low, high, fs)
        filtered = lfilter(b, a, samples)
        gain = 10 ** (gains[i] / 20)
        out += filtered * gain
    # Нормалізація після еквалайзера
    if np.max(np.abs(out)) > 0:
        out = out / np.max(np.abs(out)) * 0.95
    return out

def create_audio_plot(samples, fs, title="Аудіо сигнал"):
    """Створює графік аудіо сигналу"""
    duration = len(samples) / fs
    t = np.linspace(0, duration, len(samples))
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, samples, 'b-', linewidth=0.5)
    plt.title(title)
    plt.xlabel('Час (секунди)')
    plt.ylabel('Амплітуда')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Збереження графіка
    plot_path = f"/tmp/{title.replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path

def main():
    print("🎵 Аудіо Рекордер та Редактор v1.0.4")
    print("=" * 50)
    print("🔧 Нові функції v1.0.4:")
    print("   ✅ Покращено шумоподавлення")
    print("   ✅ Додано кнопку 'Відтворити з EQ'")
    print("   ✅ Виправлено повторення звуку")
    print("   ✅ Перекладено інтерфейс на українську")
    print("=" * 50)
    
    # Перевірка наявності аудіо файлів
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
        import glob
        audio_files.extend(glob.glob(ext))
    
    if not audio_files:
        print("📁 Не знайдено аудіо файлів у поточній директорії")
        print("💡 Додайте аудіо файл (WAV, MP3, FLAC, OGG) та запустіть знову")
        return
    
    print("📁 Знайдені аудіо файли:")
    for i, file in enumerate(audio_files, 1):
        print(f"  {i}. {file}")
    
    # Вибір файлу
    try:
        choice = int(input("\nВиберіть номер файлу: ")) - 1
        if choice < 0 or choice >= len(audio_files):
            print("❌ Неправильний вибір!")
            return
        selected_file = audio_files[choice]
    except ValueError:
        print("❌ Введіть правильний номер!")
        return
    
    print(f"\n🎵 Обробка файлу: {selected_file}")
    
    try:
        # Завантаження аудіо
        audio = AudioSegment.from_file(selected_file)
        print(f"✅ Файл завантажено")
        print(f"   📊 Тривалість: {len(audio)/1000:.2f} сек")
        print(f"   🔊 Частота: {audio.frame_rate} Гц")
        print(f"   🎧 Канали: {audio.channels}")
        
        # Конвертація в моно
        if audio.channels == 2:
            audio = audio.set_channels(1)
            print("   🔄 Конвертовано в моно")
        
        # Отримання зразків
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples = samples / np.max(np.abs(samples))  # Нормалізація
        
        # Створення графіка оригіналу
        print("\n📊 Створення графіка оригінального аудіо...")
        original_plot = create_audio_plot(samples, audio.frame_rate, "Оригінальне аудіо")
        print(f"   💾 Графік збережено: {original_plot}")
        
        # Меню обробки
        while True:
            print("\n🔧 Виберіть операцію:")
            print("  1. 🧹 Обрізка тиші")
            print("  2. 🔇 Видалення шуму (покращено в v1.0.4)")
            print("  3. 🎛️ Еквалайзер")
            print("  4. 📈 Нормалізація")
            print("  5. 🔄 Повна обробка")
            print("  6. 🎛️ Відтворити з EQ (нова функція v1.0.4)")
            print("  7. 📊 Показати графік")
            print("  8. 💾 Зберегти результат")
            print("  9. ❌ Вихід")
            
            try:
                choice = int(input("\nВаш вибір: "))
            except ValueError:
                print("❌ Введіть правильний номер!")
                continue
            
            if choice == 1:  # Обрізка тиші
                print("🧹 Обрізка тиші...")
                trimmed = trim_silence(audio, silence_thresh=audio.dBFS-16, min_silence_len=100)
                samples = np.array(trimmed.get_array_of_samples()).astype(np.float32)
                samples = samples / np.max(np.abs(samples))
                print(f"   ✅ Обрізано до {len(trimmed)/1000:.2f} сек")
                
            elif choice == 2:  # Видалення шуму (покращено)
                print("🔇 Видалення шуму (покращений алгоритм v1.0.4)...")
                reduced = nr.reduce_noise(y=samples, sr=audio.frame_rate)
                samples = reduced
                print("   ✅ Шум видалено (покращено в v1.0.4)")
                
            elif choice == 3:  # Еквалайзер
                print("🎛️ Налаштування еквалайзера:")
                gains = []
                freqs = [60, 250, 1000, 4000, 16000]
                for freq in freqs:
                    try:
                        gain = float(input(f"   {freq} Гц (-12 до +12 дБ): "))
                        gains.append(max(-12, min(12, gain)))
                    except ValueError:
                        gains.append(0)
                
                print("   🎛️ Застосування еквалайзера...")
                samples = apply_equalizer(samples, audio.frame_rate, gains)
                print("   ✅ Еквалайзер застосовано")
                
            elif choice == 4:  # Нормалізація
                print("📈 Нормалізація...")
                audio_segment = AudioSegment(
                    (samples * 32767).astype(np.int16).tobytes(),
                    frame_rate=audio.frame_rate,
                    sample_width=2,
                    channels=1
                )
                normalized = effects.normalize(audio_segment)
                samples = np.array(normalized.get_array_of_samples()).astype(np.float32)
                samples = samples / np.max(np.abs(samples))
                print("   ✅ Нормалізація завершена")
                
            elif choice == 5:  # Повна обробка
                print("🔄 Повна обробка аудіо...")
                
                # 1. Обрізка тиші
                print("   🧹 Обрізка тиші...")
                trimmed = trim_silence(audio, silence_thresh=audio.dBFS-16, min_silence_len=100)
                proc_samples = np.array(trimmed.get_array_of_samples()).astype(np.float32)
                proc_samples = proc_samples / np.max(np.abs(proc_samples))
                
                # 2. Видалення шуму (покращено)
                print("   🔇 Видалення шуму (покращений алгоритм v1.0.4)...")
                reduced = nr.reduce_noise(y=proc_samples, sr=audio.frame_rate)
                
                # 3. Нормалізація
                print("   📈 Нормалізація...")
                normed = effects.normalize(AudioSegment(
                    (reduced * 32767).astype(np.int16).tobytes(),
                    frame_rate=audio.frame_rate,
                    sample_width=2,
                    channels=1
                ))
                
                # 4. Еквалайзер (за замовчуванням)
                print("   🎛️ Застосування еквалайзера...")
                eq_samples = np.array(normed.get_array_of_samples()).astype(np.float32)
                eq_samples = eq_samples / np.max(np.abs(eq_samples))
                eq_samples = apply_equalizer(eq_samples, audio.frame_rate, [0, 0, 0, 0, 0])
                
                samples = eq_samples
                print("   ✅ Повна обробка завершена")
                
            elif choice == 6:  # Відтворити з EQ (нова функція)
                print("🎛️ Відтворення з еквалайзером (нова функція v1.0.4)...")
                print("   Налаштування еквалайзера:")
                gains = []
                freqs = [60, 250, 1000, 4000, 16000]
                for freq in freqs:
                    try:
                        gain = float(input(f"   {freq} Гц (-12 до +12 дБ): "))
                        gains.append(max(-12, min(12, gain)))
                    except ValueError:
                        gains.append(0)
                
                # Застосування еквалайзера
                eq_samples = apply_equalizer(samples, audio.frame_rate, gains)
                
                # Збереження для відтворення
                output_name = f"eq_preview_{selected_file}"
                audio_segment = AudioSegment(
                    (eq_samples * 32767).astype(np.int16).tobytes(),
                    frame_rate=audio.frame_rate,
                    sample_width=2,
                    channels=1
                )
                audio_segment.export(output_name, format=output_name.split('.')[-1])
                print(f"   ✅ Збережено для відтворення: {output_name}")
                print("   💡 Використовуйте зовнішній програвач для прослуховування")
                
            elif choice == 7:  # Показати графік
                print("📊 Створення графіка...")
                plot_path = create_audio_plot(samples, audio.frame_rate, "Оброблене аудіо")
                print(f"   💾 Графік збережено: {plot_path}")
                
            elif choice == 8:  # Зберегти результат
                print("💾 Збереження результату...")
                output_name = f"processed_{selected_file}"
                
                # Конвертація в аудіо сегмент
                audio_segment = AudioSegment(
                    (samples * 32767).astype(np.int16).tobytes(),
                    frame_rate=audio.frame_rate,
                    sample_width=2,
                    channels=1
                )
                
                # Збереження
                audio_segment.export(output_name, format=output_name.split('.')[-1])
                print(f"   ✅ Збережено як: {output_name}")
                
            elif choice == 9:  # Вихід
                print("👋 До побачення!")
                break
                
            else:
                print("❌ Неправильний вибір!")
    
    except Exception as e:
        print(f"❌ Помилка: {str(e)}")
        print("💡 Переконайтеся, що файл підтримується та не пошкоджений")

if __name__ == "__main__":
    main()
