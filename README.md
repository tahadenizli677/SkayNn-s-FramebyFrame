# 🎯 SkayNn's FramebyFrame

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
- **Tek Parça (.exe) Bağımsız:** Model, arayüz ve tüm kütüphaneler tek bir `.exe` içerisine gömülmüştür; ekstra kurulum gerektirmez.

---

## 📖 Adım Adım Kullanım Kılavuzu

Programı sorunsuz bir şekilde kullanmak için aşağıdaki adımları sırasıyla uygulayın:

### 1. Önemli Oyun İçi Ayarlar (Zorunlu)
* **Pencereli Mod:** Roblox'u **F11 (Tam Ekran)** modundan çıkarın. Oyun **Pencereli (Windowed)** veya **Çerçevesiz Tam Ekran (Borderless Windowed)** modunda olmalıdır. (Tam ekranda Windows görüntü yakalama siyah ekran görür).
* **Düşman Çerçevesi (Outline):** Arsenal oyun ayarlarında düşmanların etrafındaki neon çerçevenin açık olduğundan emin olun.

---

### 2. Programı Başlatma
1. `SkayNnsFramebyFrame.exe` dosyasına sağ tıklayın ve **"Yönetici olarak çalıştır"** (Run as Administrator) seçeneğini seçin. *(Windows'un oyuna fare girdisi göndermesini engellememesi için bu gereklidir).*
2. Programın arayüzü açılacaktır.

---

### 3. Hedef Rengini Seçme (Color Mode)
Arsenal'de düşmanların etrafındaki çerçeve rengi takıma veya haritaya göre değişebilir:
1. Menüde **`🎨 VISION & COLOR`** sekmesine geçin.
2. Alttaki **12 Renkli Hazır Palet**'ten düşmanınızın etrafındaki çerçeve rengine tıklayın (örneğin: `Neon Yellow`, `Crimson Red`, `Acid Green`, `Electric Cyan` vb.).
3. **Alternatif Otomatik Algılama:** Oyunda bir düşmana bakarken üstteki **`🎨 SAMPLE COLOR`** butonuna basarsanız, sistem ekrandaki rengi 3 saniye içinde otomatik olarak hafızaya alır.

---

### 4. Tespit Testi (Scan Overlay)
* Üst bardaki **`👁 SCAN OVERLAY`** butonuna basın.
* Oyuna dönüp bir düşmana bakın. Düşmanın üzerinde kutu ve ekran ortasında FOV çemberi görünüyor mu kontrol edin.
* Kutular çıkıyorsa tespit sorunsuz çalışıyor demektir!

---

### 5. Canlı Aimbot'u Aktif Etme ve Nişan Alma
1. Üst bardaki **`▶ START LIVE`** butonuna basın. Sağ üstteki durum rozeti yeşil renkte `● RUNNING LIVE` olacaktır.
2. Oyuna dönün.
3. Düşman ekrandaki çemberin (FOV) içine girdiğinde **E tuşuna basılı tutun**.
   - Tuşa **basılı tuttuğunuz sürece** nişangah düşmanın kafasına kilitlenir.
   - Tuşu bıraktığınız an nişan alma serbest kalır.

---

## 🎮 Kontrol Tuşları

| Tuş | Varsayılan | Açıklama |
|---|---|---|
| **Aim Key** | `E` | **Basılı tutulduğunda** hedefin kafasına kilitlenir. |
| **Toggle Key** | `HOME` | Aimbot'u anında geçici olarak kapatır/açar (`Safe` / `Armed`). |
| **Emergency Stop** | `END` | Canlı döngüyü anında tamamen durdurur. |

> 💡 **Tuşları Değiştirme:** Arayüzdeki **`🎯 AIMBOT ENGINE`** sekmesinde `[ E ]` butonuna tıklayın, buton `[ PRESS KEY... ]` olacaktır. Klavyenizden veya farenizden atamak istediğiniz tuşa (örneğin Mouse yan tuşları, Shift, Alt vb.) basmanız yeterlidir.

---

## 🛠️ Hassasiyet & İnce Ayarlar

Arayüzdeki kaydırıcıları oyun tarzınıza göre ayarlayabilirsiniz (değişiklikler anında oyuna yansır):

* **FOV Radius (Piksel):** Nişan alma çemberinin genişliğidir. Çember ne kadar genişse o kadar uzaktaki düşmanlara kilitlenir.
* **Smooth (Yumuşatma):** 
  - `0.10 - 0.20`: Çok hızlı, anında kafaya yapışan aim.
  - `0.40 - 0.50`: Yumuşak, insan gibi kayan doğal ve meşru (legit) aim.
* **Gain (Fare Çarpanı):** Roblox kamera hassasiyetinize göre farenin hareket gücünü ayarlar. (Titreme olursa gain'i hafifçe düşürün).
* **Head Height Offset:** `0.12` tam kafa/alın hizasıdır. Göğüse nişan almak isterseniz değeri `0.25` seviyesine çekebilirsiniz.
* **Min Target Area (Uzaklık):** `35 px²` olarak ayarlıdır. Haritanın en uzak köşelerindeki küçücük hedefleri bile algılar.

---

## ❓ Sık Karşılaşılan Sorunlar ve Çözümleri

**1. Ekranda düşmanların üzerinde kutu çıkmıyor:**
- Roblox'un **F11 (Tam Ekran)** modunda olmadığından emin olun (Pencereli moda alın).
- Düşmanların çerçeve rengiyle arayüzde seçtiğiniz rengin uyuştuğundan emin olun (`VISION & COLOR` sekmesinden rengi değiştirin veya `SAMPLE COLOR` yapın).

**2. Kutular çıkıyor ama E tuşuna basınca fare hareket etmiyor:**
- Programı **"Yönetici olarak çalıştır"** seçeneğiyle açtığınızdan emin olun.
- "Scan Overlay" modunda değil, **"Start Live"** modunda olduğunuzdan emin olun (Scan modu fareyi hareket ettirmez).
- Yanlışlıkla **Home** tuşuna bastıysanız bot `SAFE` moduna geçmiştir. Tekrar **Home** tuşuna basarak aktif edin.

**3. Nişangah düşmanın üzerinde titriyor ("gitgel" yapıyor):**
- Menüden `Gain` değerini biraz düşürün (örneğin `0.70` - `0.85` aralığına) veya `Smooth` değerini `0.40` civarına getirin.
- `aimbot.ini` içinde `lead = 0` olduğundan emin olun.

---

## 💻 Geliştiriciler İçin

### Kaynak Koddan Çalıştırma:
```bash
# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Çalıştırın
python aimbot.py
```

### Tek Parça (.exe) Olarak Derleme:
```bash
build.bat
```
Çıktı dosyası `dist\SkayNnsFramebyFrame.exe` olarak üretilir.
