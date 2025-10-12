# 🔨 Інструкції збірки EXE файлів

## 🚀 Автоматична збірка через GitHub Actions

### ✅ **Налаштовано автоматично:**
- При кожному push в `main` ветку
- При створенні Release
- GitHub автоматично збирає EXE файли

### 📦 **Що створюється:**
- `AudioEditor_v1.0.4.exe` - Графічна версія (PyQt5)
- `AudioEditor_Console_v1.0.4.exe` - Консольна версія

### 🔗 **Де знайти EXE файли:**
1. Перейдіть в **Actions** вкладку на GitHub
2. Виберіть останній workflow
3. Завантажте артефакти

---

## 🛠️ Локальна збірка

### **Спосіб 1: Автоматичний скрипт**
```bash
python build_exe.py
```

### **Спосіб 2: Ручна збірка**
```bash
# Встановлення PyInstaller
pip install pyinstaller

# Графічна версія (без консолі)
pyinstaller --onefile --windowed --name "AudioEditor_v1.0.4" audio_recorder_editor.py

# Консольна версія
pyinstaller --onefile --name "AudioEditor_Console_v1.0.4" audio_editor_replit.py
```

---

## 📋 **Параметри PyInstaller:**

- `--onefile` - Один EXE файл
- `--windowed` - Без консолі (для GUI)
- `--name` - Назва EXE файлу
- `--icon` - Іконка (якщо є)

---

## 🎯 **Результат:**
- EXE файли в папці `dist/`
- Розмір: ~50-100 MB кожен
- Запуск без встановлення Python
- Підтримка всіх функцій програми

---

## 🔧 **Усунення проблем:**

### **Проблема: "Module not found"**
```bash
pip install -r requirements.txt
```

### **Проблема: "PyQt5 not found"**
```bash
pip install PyQt5
```

### **Проблема: Великий розмір EXE**
- Використовуйте `--exclude-module` для невикористовуваних модулів
- Або `--onedir` замість `--onefile`

---

*Створено для проекту "Аудіо Рекордер та Редактор v1.0.4"*
