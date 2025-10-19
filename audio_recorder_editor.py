import sys
import os
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pydub import AudioSegment, effects, silence
from pydub.silence import detect_nonsilent
import pyaudio
import wave
import noisereduce as nr
from scipy.signal import butter, lfilter
from threading import Thread

# --- Функція для обрізки тиші ---
def trim_silence(audio, silence_thresh=-40, min_silence_len=100):
    nonsilent = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not nonsilent:
        return audio
    start_trim = nonsilent[0][0]
    end_trim = nonsilent[-1][1]
    return audio[start_trim:end_trim]

class AudioRecorderEditor(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Аудіо Рекордер та Редактор')
        self.setGeometry(100, 100, 900, 700)
        self.audio_data = None
        self.fs = 44100
        self.filename = None
        self.processed_audio = None
        self.is_recording = False
        self.duration = 5
        self.eq_gains = [0, 0, 0, 0, 0]  # дБ для 5 смуг
        self.history = []  # стек історії для скасування
        self.redo_stack = []  # стек для повернення
        self.selection = None  # (початок, кінець) в секундах
        self.play_thread = None
        self.dragging = False
        self.drag_start = None
        self.init_ui()
        self.setup_shortcuts()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Верхня панель: запис, тривалість, завантаження, збереження, обробка, відтворення
        top_layout = QtWidgets.QHBoxLayout()
        self.btn_record = QtWidgets.QPushButton('Записати')
        self.btn_record.clicked.connect(self.record_audio)
        top_layout.addWidget(self.btn_record)

        top_layout.addWidget(QtWidgets.QLabel('Тривалість (сек):'))
        self.spin_duration = QtWidgets.QSpinBox()
        self.spin_duration.setRange(1, 30)
        self.spin_duration.setValue(5)
        self.spin_duration.valueChanged.connect(self.set_duration)
        top_layout.addWidget(self.spin_duration)

        self.btn_load = QtWidgets.QPushButton('Завантажити файл')
        self.btn_load.clicked.connect(self.load_audio)
        top_layout.addWidget(self.btn_load)

        self.btn_save = QtWidgets.QPushButton('Зберегти')
        self.btn_save.clicked.connect(self.save_audio)
        self.btn_save.setEnabled(False)
        top_layout.addWidget(self.btn_save)

        self.btn_play = QtWidgets.QPushButton('Відтворити')
        self.btn_play.clicked.connect(self.play_audio)
        self.btn_play.setEnabled(False)
        top_layout.addWidget(self.btn_play)

        self.btn_play_eq = QtWidgets.QPushButton('Відтворити з EQ')
        self.btn_play_eq.clicked.connect(self.play_audio_with_eq)
        self.btn_play_eq.setEnabled(False)
        top_layout.addWidget(self.btn_play_eq)

        self.btn_process = QtWidgets.QPushButton('Обробити')
        self.btn_process.clicked.connect(self.process_audio)
        self.btn_process.setEnabled(False)
        top_layout.addWidget(self.btn_process)

        self.btn_undo = QtWidgets.QPushButton('Скасувати')
        self.btn_undo.clicked.connect(self.undo)
        self.btn_undo.setEnabled(False)
        top_layout.addWidget(self.btn_undo)

        self.btn_redo = QtWidgets.QPushButton('Повернути')
        self.btn_redo.clicked.connect(self.redo)
        self.btn_redo.setEnabled(False)
        top_layout.addWidget(self.btn_redo)

        layout.addLayout(top_layout)

        # Еквалайзер
        eq_layout = QtWidgets.QHBoxLayout()
        self.eq_sliders = []
        self.eq_labels = []
        self.eq_freqs = [60, 250, 1000, 4000, 16000]
        for i, freq in enumerate(self.eq_freqs):
            vbox = QtWidgets.QVBoxLayout()
            label = QtWidgets.QLabel(f'{freq} Гц')
            vbox.addWidget(label)
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
            slider.setRange(-12, 12)
            slider.setValue(0)
            slider.setTickInterval(1)
            slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBothSides)
            slider.valueChanged.connect(self.update_eq_gains)
            vbox.addWidget(slider)
            self.eq_sliders.append(slider)
            self.eq_labels.append(label)
            eq_layout.addLayout(vbox)
        layout.addLayout(eq_layout)

        # Графік
        self.plot_widget = pg.PlotWidget(title='Хвиля аудіо')
        self.plot_widget.setLabel('bottom', 'Час', units='с')
        self.plot_widget.setLabel('left', 'Амплітуда')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMouseEnabled(x=False, y=False)  # Відключаємо переміщення мишею
        layout.addWidget(self.plot_widget, stretch=2)
        # RegionItem для виділення
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.region.setVisible(False)
        self.plot_widget.addItem(self.region)
        # Шар для виділення (LinePlotItem)
        self.selection_curve = None  # буде створюватися динамічно
        # Події миші
        self.plot_widget.scene().sigMouseClicked.connect(self.on_plot_mouse_click)
        self.plot_widget.scene().sigMouseMoved.connect(self.on_plot_mouse_move)
        self.plot_widget.viewport().installEventFilter(self)
        self.plot_widget.wheelEvent = self.on_wheel_event
        self.region.sigRegionChanged.connect(self.on_region_changed)
        
        # Нижня панель: вибір джерел запису та відтворення
        bottom_layout = QtWidgets.QHBoxLayout()
        
        # Вибір джерела запису
        bottom_layout.addWidget(QtWidgets.QLabel('Джерело запису:'))
        self.input_device_combo = QtWidgets.QComboBox()
        self.input_device_combo.setMinimumWidth(200)
        self.populate_input_devices()
        bottom_layout.addWidget(self.input_device_combo)
        
        # Вибір джерела відтворення
        bottom_layout.addWidget(QtWidgets.QLabel('Джерело відтворення:'))
        self.output_device_combo = QtWidgets.QComboBox()
        self.output_device_combo.setMinimumWidth(200)
        self.populate_output_devices()
        bottom_layout.addWidget(self.output_device_combo)
        
        # Кнопка оновлення списку пристроїв
        self.btn_refresh_devices = QtWidgets.QPushButton('Оновити пристрої')
        self.btn_refresh_devices.clicked.connect(self.refresh_devices)
        bottom_layout.addWidget(self.btn_refresh_devices)
        
        bottom_layout.addStretch()  # Розтягуємо залишковий простір
        layout.addLayout(bottom_layout)
        
        self.setLayout(layout)

    def setup_shortcuts(self):
        undo_shortcut = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+Z'), self)
        undo_shortcut.activated.connect(self.undo)
        redo_shortcut = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+Shift+Z'), self)
        redo_shortcut.activated.connect(self.redo)

    def populate_input_devices(self):
        """Заповнює список пристроїв запису"""
        self.input_device_combo.clear()
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:  # Пристрій підтримує запис
                    device_name = info['name']
                    self.input_device_combo.addItem(f"{i}: {device_name}", i)
            p.terminate()
        except Exception as e:
            print(f"Помилка отримання списку пристроїв запису: {e}")
            self.input_device_combo.addItem("Помилка завантаження пристроїв", -1)

    def populate_output_devices(self):
        """Заповнює список пристроїв відтворення"""
        self.output_device_combo.clear()
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxOutputChannels'] > 0:  # Пристрій підтримує відтворення
                    device_name = info['name']
                    self.output_device_combo.addItem(f"{i}: {device_name}", i)
            p.terminate()
        except Exception as e:
            print(f"Помилка отримання списку пристроїв відтворення: {e}")
            self.output_device_combo.addItem("Помилка завантаження пристроїв", -1)

    def refresh_devices(self):
        """Оновлює списки всіх пристроїв"""
        # Зберігаємо вибрані пристрої
        selected_input = self.get_selected_input_device()
        selected_output = self.get_selected_output_device()
        
        # Оновлюємо списки
        self.populate_input_devices()
        self.populate_output_devices()
        
        # Відновлюємо вибрані пристрої, якщо вони все ще доступні
        if selected_input is not None and selected_input != -1:
            for i in range(self.input_device_combo.count()):
                if self.input_device_combo.itemData(i) == selected_input:
                    self.input_device_combo.setCurrentIndex(i)
                    break
        
        if selected_output is not None and selected_output != -1:
            for i in range(self.output_device_combo.count()):
                if self.output_device_combo.itemData(i) == selected_output:
                    self.output_device_combo.setCurrentIndex(i)
                    break

    def get_selected_input_device(self):
        """Повертає індекс вибраного пристрою запису"""
        return self.input_device_combo.currentData()

    def get_selected_output_device(self):
        """Повертає індекс вибраного пристрою відтворення"""
        return self.output_device_combo.currentData()

    def set_duration(self, val):
        self.duration = val

    def update_eq_gains(self):
        self.eq_gains = [slider.value() for slider in self.eq_sliders]

    def eventFilter(self, obj, event):
        if obj is self.plot_widget.viewport():
            if event.type() == QtCore.QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._start_selection(event)
                return True
            elif event.type() == QtCore.QEvent.Type.MouseMove and self.dragging:
                self._update_selection(event)
                return True
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease and event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._end_selection(event)
                return True
        return super().eventFilter(obj, event)

    def _start_selection(self, event):
        if self.audio_data is None:
            return
        pos = self.plot_widget.plotItem.vb.mapSceneToView(self.plot_widget.mapToScene(event.pos()))
        t = max(0, min(len(np.frombuffer(self.audio_data, dtype=np.int16)) / self.fs, pos.x()))
        self.dragging = True
        self.drag_start = t
        self.region.setRegion([t, t])
        self.region.setVisible(True)

    def _update_selection(self, event):
        if self.audio_data is None or not self.dragging:
            return
        pos = self.plot_widget.plotItem.vb.mapSceneToView(self.plot_widget.mapToScene(event.pos()))
        t = max(0, min(len(np.frombuffer(self.audio_data, dtype=np.int16)) / self.fs, pos.x()))
        self.region.setRegion([self.drag_start, t])
        self.selection = tuple(sorted([self.drag_start, t]))

    def _end_selection(self, event):
        self.dragging = False
        r = self.region.getRegion()
        if abs(r[1] - r[0]) < 0.01:
            self.region.setVisible(False)
            self.selection = None
        else:
            self.selection = tuple(sorted(r))

    def on_plot_mouse_click(self, event):
        if self.audio_data is None:
            return
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.region.setVisible(False)
            self.selection = None

    def on_plot_mouse_move(self, pos):
        pass  # обробка тепер через eventFilter

    def on_wheel_event(self, event):
        vb = self.plot_widget.getViewBox()
        delta = event.angleDelta().y()
        vb.scaleBy((0.9 if delta > 0 else 1.1, 1))

    def on_region_changed(self):
        if self.region.isVisible():
            r = self.region.getRegion()
            self.selection = tuple(sorted(r))

    def record_audio(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.btn_record.setText('Запис...')
        self.audio_data = []
        self.record_thread = Thread(target=self._record_thread)
        self.record_thread.start()
        # Використовуємо QTimer для зупинки запису
        self.record_timer = QtCore.QTimer()
        self.record_timer.timeout.connect(self.stop_recording)
        self.record_timer.setSingleShot(True)
        self.record_timer.start(self.duration * 1000)

    def _record_thread(self):
        p = pyaudio.PyAudio()
        input_device = self.get_selected_input_device()
        if input_device is None or input_device == -1:
            input_device = None  # Використовуємо пристрій за замовчуванням
        
        try:
            stream = p.open(
                format=pyaudio.paInt16, 
                channels=1, 
                rate=self.fs, 
                input=True, 
                input_device_index=input_device,
                frames_per_buffer=1024
            )
            frames = []
            for _ in range(0, int(self.fs / 1024 * self.duration)):
                data = stream.read(1024)
                frames.append(data)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Помилка запису з вибраним пристроєм: {e}")
            # Пробуємо з пристроєм за замовчуванням
            try:
                stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.fs, input=True, frames_per_buffer=1024)
                frames = []
                for _ in range(0, int(self.fs / 1024 * self.duration)):
                    data = stream.read(1024)
                    frames.append(data)
                stream.stop_stream()
                stream.close()
            except Exception as e2:
                print(f"Помилка запису з пристроєм за замовчуванням: {e2}")
                frames = []
        finally:
            p.terminate()
        
        if frames:
            self.audio_data = b''.join(frames)  # raw int16 без нормалізації
            self.update_plot()
            self.btn_save.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.btn_play_eq.setEnabled(True)
            self.btn_process.setEnabled(True)
            self.history.clear()
            self.redo_stack.clear()
            self.btn_undo.setEnabled(False)
            self.btn_redo.setEnabled(False)

    def stop_recording(self):
        self.is_recording = False
        self.btn_record.setText('Записати')
        # Зупиняємо таймер якщо він активний
        if hasattr(self, 'record_timer') and self.record_timer.isActive():
            self.record_timer.stop()

    def load_audio(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Відкрити аудіофайл', '', 'Аудіо файли (*.wav *.mp3 *.flac *.ogg)')
        if fname:
            audio = AudioSegment.from_file(fname)
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            self.audio_data = samples.astype(np.int16).tobytes()
            self.fs = audio.frame_rate
            self.filename = fname
            self.update_plot()
            self.btn_save.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.btn_play_eq.setEnabled(True)
            self.btn_process.setEnabled(True)
            self.history.clear()
            self.redo_stack.clear()
            self.btn_undo.setEnabled(False)
            self.btn_redo.setEnabled(False)

    def save_audio(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Зберегти аудіофайл', '', 'WAV (*.wav);;MP3 (*.mp3)')
        if fname:
            ext = os.path.splitext(fname)[1].lower()
            if self.processed_audio is not None:
                audio = self.processed_audio
            else:
                audio = AudioSegment(
                    self.audio_data,
                    frame_rate=self.fs,
                    sample_width=2,
                    channels=1
                )
            audio.export(fname, format=ext[1:])

    def process_audio(self):
        if self.audio_data is None:
            return
        self.history.append(self.audio_data)
        self.btn_undo.setEnabled(True)
        self.redo_stack.clear()
        self.btn_redo.setEnabled(False)
        samples = np.frombuffer(self.audio_data, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0
        # Визначаємо діапазон обробки
        if self.region.isVisible() and self.selection:
            start_sec, end_sec = self.selection
            start_idx = int(start_sec * self.fs)
            end_idx = int(end_sec * self.fs)
            if start_idx == end_idx:
                return
            to_process = samples[start_idx:end_idx]
            # 1. Обрізка тиші
            audio = AudioSegment(
                (to_process * 32768).astype(np.int16).tobytes(),
                frame_rate=self.fs,
                sample_width=2,
                channels=1
            )
            trimmed = trim_silence(audio, silence_thresh=audio.dBFS-16, min_silence_len=100)
            proc_samples = np.array(trimmed.get_array_of_samples()).astype(np.float32)
            proc_samples = proc_samples / 32768.0
            # 2. Видалення шуму
            reduced = nr.reduce_noise(y=proc_samples, sr=self.fs)
            # 3. Нормалізація (тільки один раз, без повторного масштабування)
            normed = effects.normalize(AudioSegment(
                (reduced * 32768).astype(np.int16).tobytes(),
                frame_rate=self.fs,
                sample_width=2,
                channels=1
            ))
            eq_samples = np.array(normed.get_array_of_samples()).astype(np.float32)
            eq_samples = self.apply_equalizer(eq_samples, self.fs, self.eq_gains)
            # Вставляємо назад, коректно по довжині
            out = samples.copy()
            n = min(len(eq_samples), end_idx - start_idx)
            out[start_idx:start_idx+n] = eq_samples[:n]
            self.audio_data = (out * 32768).clip(-32768, 32767).astype(np.int16).tobytes()
        else:
            audio = AudioSegment(
                (samples * 32768).astype(np.int16).tobytes(),
                frame_rate=self.fs,
                sample_width=2,
                channels=1
            )
            trimmed = trim_silence(audio, silence_thresh=audio.dBFS-16, min_silence_len=100)
            samples = np.array(trimmed.get_array_of_samples()).astype(np.float32)
            samples = samples / 32768.0
            reduced = nr.reduce_noise(y=samples, sr=self.fs)
            normed = effects.normalize(AudioSegment(
                (reduced * 32768).astype(np.int16).tobytes(),
                frame_rate=self.fs,
                sample_width=2,
                channels=1
            ))
            eq_samples = np.array(normed.get_array_of_samples()).astype(np.float32)
            eq_samples = self.apply_equalizer(eq_samples, self.fs, self.eq_gains)
            self.audio_data = (eq_samples * 32768).clip(-32768, 32767).astype(np.int16).tobytes()
        self.update_plot()

    def undo(self):
        if not self.history:
            return
        self.redo_stack.append(self.audio_data)
        self.audio_data = self.history.pop()
        self.update_plot()
        self.btn_redo.setEnabled(True)
        if not self.history:
            self.btn_undo.setEnabled(False)

    def redo(self):
        if not self.redo_stack:
            return
        self.history.append(self.audio_data)
        self.audio_data = self.redo_stack.pop()
        self.update_plot()
        self.btn_undo.setEnabled(True)
        if not self.redo_stack:
            self.btn_redo.setEnabled(False)

    def butter_bandpass(self, lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_equalizer(self, samples, fs, gains):
        # 5 смуг: 60, 250, 1000, 4000, 16000 Гц
        bands = [(20, 120), (120, 500), (500, 2000), (2000, 8000), (8000, 20000)]
        out = np.zeros_like(samples)
        for i, (low, high) in enumerate(bands):
            b, a = self.butter_bandpass(low, high, fs)
            filtered = lfilter(b, a, samples)
            gain = 10 ** (gains[i] / 20)
            out += filtered * gain
        # Нормалізація після еквалайзера (без множення на 32767)
        if np.max(np.abs(out)) > 0:
            out = out / np.max(np.abs(out)) * 0.95
        return out

    def play_audio(self):
        if self.audio_data is None:
            return
        if self.play_thread and self.play_thread.is_alive():
            return
        # Відтворюємо оригінальні дані без додаткової обробки еквалайзером
        # Еквалайзер вже застосований до self.audio_data при обробці
        self.play_thread = Thread(target=self._play_thread, args=(self.audio_data,))
        self.play_thread.start()

    def play_audio_with_eq(self):
        if self.audio_data is None:
            return
        if self.play_thread and self.play_thread.is_alive():
            return
        # Застосовуємо еквалайзер до аудіо для відтворення в реальному часі
        samples = np.frombuffer(self.audio_data, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0
        eq_samples = self.apply_equalizer(samples, self.fs, self.eq_gains)
        play_data = (eq_samples * 32768).clip(-32768, 32767).astype(np.int16).tobytes()
        self.play_thread = Thread(target=self._play_thread, args=(play_data,))
        self.play_thread.start()

    def _play_thread(self, play_data):
        try:
            p = pyaudio.PyAudio()
            output_device = self.get_selected_output_device()
            if output_device is None or output_device == -1:
                output_device = None  # Використовуємо пристрій за замовчуванням
            
            try:
                stream = p.open(
                    format=pyaudio.paInt16, 
                    channels=1, 
                    rate=self.fs, 
                    output=True,
                    output_device_index=output_device
                )
                stream.write(play_data)
                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"Помилка відтворення з вибраним пристроєм: {e}")
                # Пробуємо з пристроєм за замовчуванням
                stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.fs, output=True)
                stream.write(play_data)
                stream.stop_stream()
                stream.close()
        except Exception as e:
            print(f"Помилка відтворення: {e}")
        finally:
            p.terminate()
            # Переконуємося, що потік завершений
            self.play_thread = None

    def update_plot(self):
        self.plot_widget.clear()
        if self.audio_data is not None:
            samples = np.frombuffer(self.audio_data, dtype=np.int16)
            t = np.arange(len(samples)) / self.fs
            self.plot_widget.plot(t, samples, pen='b')
            # Візуалізація виділення зеленим кольором
            if self.selection and self.region.isVisible():
                start_sec, end_sec = self.selection
                start_idx = int(start_sec * self.fs)
                end_idx = int(end_sec * self.fs)
                if end_idx > start_idx:
                    sel_t = t[start_idx:end_idx]
                    sel_samples = samples[start_idx:end_idx]
                    if self.selection_curve is not None:
                        self.plot_widget.removeItem(self.selection_curve)
                    self.selection_curve = pg.PlotCurveItem(sel_t, sel_samples, pen=pg.mkPen('g', width=2))
                    self.plot_widget.addItem(self.selection_curve)
                else:
                    if self.selection_curve is not None:
                        self.plot_widget.removeItem(self.selection_curve)
                        self.selection_curve = None
                self.region.setRegion(self.selection)
                self.region.setVisible(True)
            else:
                if self.selection_curve is not None:
                    self.plot_widget.removeItem(self.selection_curve)
                    self.selection_curve = None
                self.region.setVisible(False)
            self.plot_widget.setXRange(0, max(t) if len(t) > 0 else 1)
            self.plot_widget.setYRange(-32768, 32767)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = AudioRecorderEditor()
    win.show()
    sys.exit(app.exec())