import tkinter as tk
from tkinter import filedialog, messagebox
import os
from pydub import AudioSegment, effects, silence
import numpy as np
import noisereduce as nr

# Проверка наличия ffmpeg для работы с mp3
AudioSegment.converter = 'ffmpeg' if os.name != 'nt' else 'ffmpeg.exe'

class AudioCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Audio Cleaner')
        self.audio = None
        self.processed_audio = None
        self.filename = None

        self.load_btn = tk.Button(root, text='Загрузить аудиофайл', command=self.load_audio)
        self.load_btn.pack(pady=10)

        self.process_btn = tk.Button(root, text='Обработать', command=self.process_audio, state=tk.DISABLED)
        self.process_btn.pack(pady=10)

        self.save_btn = tk.Button(root, text='Сохранить результат', command=self.save_audio, state=tk.DISABLED)
        self.save_btn.pack(pady=10)

    def load_audio(self):
        filetypes = [('Audio Files', '*.wav *.mp3 *.flac *.ogg'), ('All Files', '*.*')]
        filename = filedialog.askopenfilename(title='Выберите аудиофайл', filetypes=filetypes)
        if filename:
            try:
                self.audio = AudioSegment.from_file(filename)
                self.filename = filename
                self.process_btn.config(state=tk.NORMAL)
                messagebox.showinfo('Успех', f'Файл {os.path.basename(filename)} загружен!')
            except Exception as e:
                messagebox.showerror('Ошибка', f'Не удалось загрузить файл: {e}')

    def process_audio(self):
        if not self.audio:
            messagebox.showerror('Ошибка', 'Сначала загрузите аудиофайл!')
            return
        try:
            # 1. Обрезка тишины
            trimmed = self.trim_silence(self.audio)
            # 2. Конвертация в numpy для noisereduce
            samples = np.array(trimmed.get_array_of_samples()).astype(np.float32)
            if trimmed.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)  # моно
            samples = samples / np.iinfo(trimmed.array_type).max
            # 3. Удаление шума
            reduced_noise = nr.reduce_noise(y=samples, sr=trimmed.frame_rate)
            # 4. Нормализация
            normed = effects.normalize(AudioSegment(
                (reduced_noise * np.iinfo(trimmed.array_type).max).astype(trimmed.array_type).tobytes(),
                frame_rate=trimmed.frame_rate,
                sample_width=trimmed.sample_width,
                channels=1
            ))
            self.processed_audio = normed
            self.save_btn.config(state=tk.NORMAL)
            messagebox.showinfo('Готово', 'Обработка завершена!')
        except Exception as e:
            messagebox.showerror('Ошибка', f'Ошибка обработки: {e}')

    def trim_silence(self, audio, silence_thresh=-40, chunk_size=10):
        # Обрезка тишины в начале и конце
        dBFS = audio.dBFS
        silence_thresh = dBFS + silence_thresh
        trimmed = silence.strip_silence(audio, silence_thresh=silence_thresh, chunk_size=chunk_size)
        return trimmed

    def save_audio(self):
        if not self.processed_audio:
            messagebox.showerror('Ошибка', 'Нет обработанного аудио для сохранения!')
            return
        filetypes = [('WAV', '*.wav'), ('MP3', '*.mp3'), ('FLAC', '*.flac')]
        save_path = filedialog.asksaveasfilename(defaultextension='.wav', filetypes=filetypes)
        if save_path:
            try:
                ext = os.path.splitext(save_path)[1].lower()
                self.processed_audio.export(save_path, format=ext[1:])
                messagebox.showinfo('Успех', f'Файл сохранён: {save_path}')
            except Exception as e:
                messagebox.showerror('Ошибка', f'Не удалось сохранить файл: {e}')

if __name__ == '__main__':
    root = tk.Tk()
    app = AudioCleanerApp(root)
    root.mainloop()
