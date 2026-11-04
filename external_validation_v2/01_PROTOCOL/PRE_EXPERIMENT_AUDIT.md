# Ön denetim: ilk dış deneyin kodu ve sonuçları

**Kapsam.** `EnvSAX_EXTERNAL_VALIDATION/` (2026-11-01) kodu ve çıktıları V2'den önce okundu. Bu belge o incelemenin dökümüdür.

**Zamanlama.** İnceleme V2 protokolü kilitlenmeden (2026-11-02 13:39) önce yapıldı. Belge sonradan yazıldı. İçerik yalnız ilk deneyin dosyalarına dayanır. Eski klasörde hiçbir dosya değiştirilmedi.

## 1. İncelenen bileşenler

| Bileşen | Dosya | Bulgu |
|---|---|---|
| Katılımcı manifestosu | `02_DATA/dryad_shor_benninger/participant_manifest.csv`, `dryad_frozen/participant_manifest_analysis.csv` | 43 AH, 96 kontrol. Hepsinde ≥ 102 temiz pencere var, dışlanan yok. AH'de yalnız ilk kayıt kullanılıyor. |
| Ön işleme | `code/02_prepare_dryad.py` | Baştaki ve sondaki tümü-sıfır örnekler kesiliyor, 5 s kenar kırpılıyor, 500 Hz → 200 Hz (MNE `npad="auto"`). Filtre ve Hilbert sürekli kayıtta, sonra 10 s / %50 pencere (`evlib.encode_continuous`). AHEPA `01_preprocess.py` ile aynı yol. |
| Kanal işleme | `02_prepare_dryad.py`, `03_frozen_fidelity.py` | Kanal adı atanmıyor, sütunlar ch01–ch19. Etiketsiz tutarlılık denetimi `channel_consistency.csv`'de. |
| Sinyal birimi | `03_frozen_fidelity.py` (birim kuralı), `EXTERNAL_PROTOCOL.md` §4, §11.1 | Yayın "[mV]" diyor. Katılımcılar arası medyan geniş bant RMS 7.90 olduğu için µV kabul edilmiş. Kural sonuçtan önce yazılmış. |
| Bant gücü | `evlib.encode_continuous` | `log10(mean(x_b²) + 1e-12)`, µV². Göreli güç doğrusal ölçekte 5 bant üzerinden normalize. |
| Kuantil kalibrasyonu | `symbolic.PowerQuantizer`, `00_build_frozen_encoder.py` | Bant başına 7 ampirik kuantil kesim noktası. AHEPA'da 39 fit katılımcısı, katılımcı başına 25 pencere. |
| Ölçek kod çözücü | `evlib.FrozenScale.decode` | Seviye içi kalibrasyon ortalaması (temsil değeri). |
| Cohen's d | `common.cohens_d` | Birleşik SD, `ddof=1`. Profil katılımcı-ortalama göreli güçten hesaplanıyor (95 değer). |
| r_d, CCC, bant ortalaması çıkarılmış r_d | `evlib.fidelity` | r_d = 95 değer üzerinde Pearson r. CCC = Lin, `ddof=0`. Bant ortalaması çıkarılmış r_d = her bantta 19 kanal ortalaması çıkarıldıktan sonra Pearson r. |
| Bootstrap | `evlib.boot_fidelity` | 2.000 tekrar. Grup içi tabakalı katılımcı yeniden örneklemesi, seed 49. Özgün ve çözülmüş aynı örnekte. |
| Kanal karıştırma referansı | `evlib.channel_shuffle_null` | Kanal sırası bant içinde karıştırılıyor, 10.000 permütasyon. |

## 2. Sayıların doğrulanması

Sonuç dosyaları ile makale (`EnvSAX_Manuscript_EN_LIT_EXTCHECK.docx`) karşılaştırıldı. **Hepsi uyumlu.**

| Değer | Sonuç dosyası | Makale |
|---|---|---|
| AHEPA r_d (65) | `T2_fidelity_summary.csv`: 0.97809 | 0.978 [0.903, 0.989] |
| AHEPA OOF r_d | `T2_fidelity_summary.csv`: 0.97377 | 0.974 [0.896, 0.987] |
| AHEPA bant ortalaması çıkarılmış r_d | `RESULTS_COMPARISON.csv`: 0.950 | 0.950 (OOF 0.938) |
| AHEPA held-out 26 (14 AH / 12 HC) | `ahepa_heldout26_fidelity.csv`: r_d 0.98679, CCC 0.98643, BMR 0.94591 | Tablo A1: 0.987 / 0.986 / 0.946 |
| Dryad dondurulmuş | `dryad_frozen/summary_long.csv`: r_d 0.80373 [0.309, 0.910], CCC 0.68101, BMR 0.11828 | 0.804 / 0.681 / 0.118 |
| Dryad doygunluk | `shift_diagnostics.csv`: δ 99.68, θ 94.77, α 67.76, β 71.13, γ 67.87 % | 99.7 / 94.8 / 67.8 / 71.1 / 67.9 % |
| Dryad S4 yeniden uyum | `summary_long.csv`: r_d 0.99185, CCC 0.99160, BMR 0.88706 | 0.992 / 0.992 / 0.887 |
| Dryad sınıflandırma | `classify_summary_dryad.csv` | M1 0.772, M1s 0.789, M1s-frozen 0.718 |

Yeniden hesaplama da yapıldı. V2 kodu C1 ve C2 değerlerini 1e-9 hassasiyetle yeniden üretiyor (`DECISION.json` → `checks.reproduction`). İlk bootstrap GA'sı da aynen çıkıyor: C1 r_d alt sınırı 0.30924.

## 3. Soru: r_d = 0.992 tüm dış kohortun etiketsiz kayıtlarından mı öğrenildi?

**Evet.** `03_frozen_fidelity.py` içindeki `Calibrated(lps)` sınıfı şöyle çalışıyor:

- `lps`, dahil edilen 139 katılımcının tamamının QC'den geçen pencereleri. Liste `load(man)` ile alınıyor.
- Her katılımcıdan `default_rng(42)` ile 25 pencere seçiliyor.
- Kesim noktaları ve temsil değerleri bu pencerelerden öğreniliyor.
- Fonksiyon etiket almıyor. Etiket yalnız sonradan `summarize(…, y)` içinde Cohen's d için kullanılıyor.
- Aynı kod defteri aynı 139 katılımcının bütün pencerelerine uygulanıp r_d hesaplanıyor.

Sonuç: **kalibrasyon etiketsizdi, ama kalibrasyon ve değerlendirme katılımcıları aynıydı.** Değerlendirilen her katılımcının kendi pencereleri kod defterinin öğrenilmesine katkı verdi (25 pencere / katılımcı).

## 4. Üç protokolün ayrımı

| | A. Dondurulmuş AHEPA kod defteri (C1) | B. Tüm kohort etiketsiz yeniden uyum (C2) | C. Katılımcı-ayrık etiketsiz kalibrasyon (C3, V2) |
|---|---|---|---|
| Kod defterinin öğrenildiği veri | AHEPA'nın 39 katılımcısı | Dryad'ın 139 katılımcısının hepsi | Dryad'da her kat için diğer 4 katın katılımcıları (111–112) |
| Değerlendirilen katılımcılar | Dryad 139 | Dryad 139 | Dryad 139, her biri kendisini görmemiş bir kod defteriyle |
| Hedef etiketi kalibrasyonda | Hayır | Hayır | Hayır (yalnız bölmenin tabakalanmasında) |
| Değerlendirilen katılımcı kalibrasyonda | Hayır | **Evet** | Hayır |
| Neyi test eder | Merkezler arası dondurulmuş aktarım | Kohorta özgü kod defterinin aynı örneklemdeki sadakati | Kohorta özgü etiketsiz kod defterinin aynı kohortun yeni katılımcılarına aktarılabilirliği |
| Neyi test etmez | – | Yeni katılımcıya genelleme | Merkezler arası dondurulmuş aktarım; klinik genellenebilirlik |

## 5. V2'yi etkileyen ek gözlemler

- **AHEPA makale OOF r_d'si (0.974), V2'den farklı bir pencere protokolüyle hesaplandı.** `02_fidelity.py` → `U.quantize` eğitim katının **bütün** segmentlerini kullanıyor. C3 ise katılımcı başına 25 pencere kullanıyor; bu, dondurulmuş kodlayıcı ve S4 ile aynı. Fark raporlarda belirtildi.
- **Örnekleme hızı grupla kısmen karışık.** 200 Hz kayıtların hepsi kontrollere ait (26/96), AH kayıtlarının hepsi 500 Hz. Kalibrasyon kodlayıcıları iki hızı birlikte içeriyor. Bu durum Cohen's d profilinin kendisini etkileyebilir, kod defterini değil. İlk deneyde S6 ile ele alındı.
- **Yaş farkı belirgin.** AH medyanı 71 yaş (55–87), kontrol medyanı 53.5 yaş (19–80).
