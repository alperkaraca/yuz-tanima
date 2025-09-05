Yüz Tespiti, Takibi, Sayımı ve Isı Haritası (Kimliksiz)

Bu proje videolardan kimlik tanıma yapmadan yüzleri tespit eder, takip eder, anlık/toplam sayım metrikleri üretir, ısı haritası çıkarır ve opsiyonel olarak yüzleri bulanıklaştırır. Gerçek-zamana yakın çalışması hedeflenmiştir; GPU varsa hızlandırma, yoksa CPU üzerinde optimize akış desteklenir.

Önemli: Bu proje kimlik tanıma/face recognition YAPMAZ. Sadece tespit ve seans içi anonim ID takibidir.

Kurulum
- Python 3.10+
- Bağımlılıklar: `pip install -r requirements.txt`

Hızlı Başlat
- Giriş videosunu `data/input_videos/` klasörüne koyun veya tam yol verin.
- Komut:
  `python -m src.main --source data/input_videos/sample.mp4 --detector retinaface --tracker ocsort --blur on --heatmap on --save-annotated data/outputs/out.mp4 --save-metrics data/outputs/metrics.json`

Basit Arayüz (Streamlit)
- Kurulum sonrası arayüzü başlatın: `streamlit run src/ui/app.py`
- Tarayıcıda açılan sayfadan videonuzu yükleyin, seçenekleri belirleyip çalıştırın.
- Çıktılar `data/outputs/` klasörüne kaydedilir ve arayüzden indirilebilir.

Özellikler
- Yüz tespiti: RetinaFace/MTCNN adapterleri (paket yoksa otomatik Haar Cascade yedek)
- Takip: OC-SORT/DeepSORT adapterleri (paket yoksa basit SORT benzeri yedek)
- Çıktılar: işlenmiş video, JSON/CSV metrikler, ısı haritası PNG, canlı overlay
- Gizlilik: Gauss bulanıklaştırma; kimlik eşleştirme/embedding yok

Performans İpuçları
- `--frame-skip 1` veya `2` ile her N karede tespit yapın
- `--half-res on` ile dahili işlemede yarı çözünürlük
- GPU varsa OpenCV CUDA derlemesi/ONNX Runtime ile hızlandırma (opsiyonel)

Etik ve Hukuki Not
- Bu proje eğitim/araştırma amaçlıdır. Kamusal alan görüntülerinde yerel yasa ve etik kurallara uyun. Kimlik tespiti, veri tabanı eşleştirme yasaktır.

Testler
- Basit birim testleri `tests/` klasöründe. Çalıştırmak için: `python -m pytest -q` (pytest kuruluysa)

Yapı
- `src/detectors/`: Tespit adapterleri
- `src/trackers/`: Takip adapterleri
- `src/utils/`: Çizim, video IO, ısı haritası, metrikler, fps, gizlilik
- `src/pipeline.py`: Uçtan uca akış
- `src/main.py`: CLI
- `src/ui/app.py`: Streamlit arayüzü
- `configs/default.yaml`: Örnek konfig

Örnek Config Alanları
- score_threshold, nms_threshold, min_face_size, blur_level, frame_skip, half_res

Kabul Kriterleri
- Yerel .mp4 ile çalıştırıldığında işlenmiş video, metrik JSON/CSV ve ısı haritası PNG üretilir. Blur=on yüzleri görünür biçimde anonimleştirir.

Repo Temizliği (Veri Takibi Yok)
- `.gitignore` içinde giriş videoları ve tüm çıktı dosyaları ignore edilir:
  - `data/input_videos/*` ve `data/outputs/*` versiyon kontrolüne girmez.
  - Klasörlerin repo’da kalması için `.gitkeep` dosyaları eklenmiştir.
