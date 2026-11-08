# 🎯 Arsenal Apex Aimbot

Roblox Arsenal için geliştirilmiş yüksek performanslı, yapay zeka (YOLOv8) ve renk algılama (Neon Outline) destekli modern nişan alma asistanı.

---

## ✨ Özellikler

- **Ultra-Premium Cyberpunk Arayüz:** CustomTkinter ile tasarlanmış modern karanlık tema, canlı durum göstergesi ve dahili terminal konsolu.
- **İki Farklı Tespit Motoru:**
  - **Color Mode (Hafif & Hızlı):** Düşmanların etrafındaki neon çerçeveyi (outline) GPU gerektirmeden 2-3 milisaniyede algılar.
  - **YOLOv8 Mode (Yapay Zeka):** `best.onnx` modelini kullanarak karakter tespiti yapar.
- **Uzun Mesafe Desteği:** Haritanın en uzak noktalarındaki küçük düşman silüetlerini bile yakalayan adaptif filtreleme.
- **Anti-Jitter & Kafa Kilitlenmesi:** Titremeleri engelleyen EMA filtresi, 3 px deadzone ve doğrudan kafa yüksekliğine (`head_frac = 0.12`) kilitlenme.
- **Tek Tuşla Renk Paleti:** 12 adet hazır neon rengi tek tıkla seçebilme veya renk tekerleğiyle özel renk belirleme.
- **Tek Parça (.exe) Derleme:** Model, arayüz ve tüm kütüphaneler tek bir bağımsız çalıştırılabilir dosyaya paketlenir.

---

## 🚀 Hızlı Başlangıç

### 1. Python ile Çalıştırma
```bash
# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Programı başlatın
python aimbot.py
```

### 2. Tek Tıkla (.exe) Derleme
Projeyi bağımsız tek bir `.exe` haline getirmek için:
- `build.bat` dosyasını çift tıklayıp çalıştırın.
- Çıktı `dist\ArsenalAimbot.exe` olarak hazır hale gelir.

---

## 🎮 Kullanım ve Tuşlar

1. `ArsenalAimbot.exe` dosyasını **Yönetici olarak çalıştırın**.
2. **Start Live** butonuna basın (durum `RUNNING: LIVE` olur).
3. Oyunda düşmana doğru bakarken **E** tuşuna basılı tutun (varsayılan aim tuşu).

| Tuş | Varsayılan | Açıklama |
|---|---|---|
| **Aim Key** | `E` | Basılı tutulduğunda hedef kilitlenir. |
| **Toggle Key** | `HOME` | Aimbot'u anında pasif / aktif (Safe/Armed) yapar. |
| **Exit** | `END` | Döngüyü tamamen durdurur. |

> **Not:** Tuşları programın arayüzündeki **AIMBOT ENGINE** sekmesinden tek tıkla değiştirebilirsiniz.

---

## ⚙️ Yapılandırma (`aimbot.ini`)

Arayüz üzerinden yapılan tüm ayarlar anında `aimbot.ini` dosyasına kaydedilir ve canlı olarak uygulanır:

```ini
[aim]
key = e            # Kilitlenme tuşu
toggle_key = home  # Açma/kapama tuşu
fov = 350          # Piksel cinsinden algılama yarıçapı
smooth = 0.40      # Yumuşatma oranı (0 = anında kilitlen, 1 = yavaş kay)
gain = 0.85        # Fare hassasiyeti çarpanı
head_frac = 0.12   # Kafa yüksekliği (0.12 = tam kafa hizası)

[detect]
mode = color       # color veya yolo
min_area = 35      # Minimum hedef boyutu (uzaktaki adamlar için)
```

---

## 🧠 Model Eğitimi (Opsiyonel - YOLO Mode)

Kendi YOLOv8 modelinizi eğitmek isterseniz:

```bash
pip install -r requirements-train.txt
python fetch_dataset.py --key ROBOFLOW_KEY
python train.py
```
Eğitilen yeni `best.onnx` dosyasını proje dizinine koymanız yeterlidir.
