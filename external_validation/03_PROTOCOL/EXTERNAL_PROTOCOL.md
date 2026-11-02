# Dış Doğrulama Protokolü (önceden sabitlenmiş)

**Sabitlenme tarihi:** 2026-11-01. Dryad EEG dosyaları indirilmeden ve hiçbir hedef sinyal görülmeden önce yazıldı. Hedef veriyi gördükten sonra yapılacak her değişiklik §11'e tarih ve gerekçeyle eklenecek. Özgün metin silinmeyecek.

**Dondurulmuş kodlayıcı:** `frozen_encoder_ahepa.npz`, sha256 `05590c3a…1921`.

## 1. Araştırma sorusu ve birincil ölçüt

**Soru:** AHEPA'da öğrenilmiş ve dondurulmuş EnvSAX ölçek kodu, bağımsız bir klinik kohortta AH–kontrol grup farklarının kanal–bant etki büyüklüğü profilini koruyor mu?

**Birincil ölçüt:**
- r~d~ = Pearson(d~özgün~, d~çözülmüş~).
- 95 kanal–bant çifti; Cohen d doğrusal göreli güç üzerinden.
- Tanım `SOURCE_PIPELINE_AUDIT.md` §4'te; kaynak kodla birebir aynı.

**Birincil referans:** AHEPA'nın nicemleyici eğitiminde kullanılmayan 26 katılımcısı, aynı dondurulmuş kodlayıcıyla.
- r~d~ = 0.987 [0.955, 0.992].
- Makaledeki 0.978 (65 katılımcı) ve katlar dışı 0.974 (her katta yeni nicemleyici) farklı protokollerdir; tabloda ayrı satırlarda verilecek.

## 2. Veri seti sırası

1. **Dryad Benninger/Shor** (doi:10.5061/dryad.8gtht76pw, v6, CC0): `alz_c1_new.mat` (AH) ve `controls_c1_new.mat`.
   - MCI, depresyon ve şizofreni dosyaları kullanılmaz ve indirilmez.
2. **Figshare, OSF FSU ve OSF ADFSU:** Uygunluk raporunda Düzey C bulundu; dış doğrulamada kullanılmayacak (`DATASET_COMPATIBILITY_REPORT_TR.md`).
3. **Dryad kullanılamazsa:** Yeni aday aranır. Bulunamazsa karar "No compatible independently accessible dataset" olur.

İlk uygun veri setinin birincil sonucu alındıktan sonra daha yüksek r~d~ için başka veri setine geçilmez.

## 3. Katılımcı ve grup kuralları

- **AH grubu:** `alz` dosyasındaki bütün katılımcılar. Yayına göre n = 40, NIA-AA ölçütleri. Dosyada MCI işareti bulunursa o katılımcı dışlanır.
- **Kontrol grubu:** `controls` dosyasındaki bütün katılımcılar. Yayına göre n = 95; nöropsikiyatrik tanı yok, rutin EEG.
- **Bağımsızlık:** AHEPA (Selanik, Yunanistan) ile Dryad (Rabin Tıp Merkezi, İsrail) ayrı merkezlerdir; katılımcı örtüşmesi olası değildir. Kanıt ve kısıtlar uygunluk raporunda.
- **Katılımcı dahil etme:** En az **6 temiz 10-s pencere** (60 s) gerekir. Dışlanan her katılımcı nedeniyle `participant_manifest.csv`'ye yazılır.
- **Yaş karıştırıcısı:**
  - Yayına göre AH 72.6 ± 7.9 yaş, kontroller 52.2 ± 16.8 yaş.
  - Birincil analiz bütün kontrollerle yapılır.
  - Önceden belirlenmiş ikincil analiz, yalnızca **60 yaş ve üstü** kontrollerle yapılır. Sınır, yayındaki AH alt sınırıdır ve sonuçtan bağımsız seçilmiştir.
  - Bu analiz ancak dosyada katılımcı yaşı varsa yapılır; yoksa yapılamadığı raporlanır.
  - Profil yaşlanma etkisini de içerebileceği için "AH'ye özgü" diye yorumlanmayacak; "AH–kontrol farkı" olarak raporlanacak.

## 4. Kayıt yapısı, kanal, birim ve göz durumu

- **Kanallar:**
  - Kanal adları dosyada ya da yazar README'sinde açıkça verilmişse AHEPA sırasına eşlenir.
  - Eşdeğer adlar (T7/T3, T8/T4, P7/T5, P8/T6) montaj tanımı doğrulanarak eşleştirilir.
  - **Kanal sırası belgelenmemişse tahmin edilmez.** Bu durumda kanal-bant profili kurulamaz; veri seti Düzey C sayılır.
  - Eksik kanal sıfırla doldurulmaz ve başka kanallardan üretilmez.
- **Birim:**
  - Belgelenmişse µV'ye çevrilir.
  - Belgelenmemişse geniş bant RMS'nin katılımcı medyanı raporlanır. 5–150 µV aralığındaysa µV kabul edilir ve "birim belgelenmemiş, genlikten çıkarıldı" diye işaretlenir.
  - Aralık dışındaysa birim "çözülemedi" sayılır. Dondurulmuş kodlayıcıya girdi verilmeden önce bu durum raporlanır.
  - **Birim ölçeklemesi hiçbir zaman r~d~'ye bakılarak seçilmez.**
- **Göz durumu:**
  - Göz kapalı bölümler dosyada işaretliyse birincil analiz yalnız bu bölümlerle yapılır.
  - İşaretli değilse sinyal örüntüsünden (ör. alfa) **tahmin edilmez**. Dağıtılan kaydın tamamı "karışık göz açık/kapalı rutin dinlenim EEG" olarak kullanılır ve protokol **Düzey B (uyarlanmış koşul)** diye etiketlenir.
- **Referans ve filtre:**
  - Yayına göre 50 Hz şebeke filtresi ve 1 Hz yüksek geçiren filtre uygulanmış; referans belirtilmemiş.
  - Ek yeniden referanslama yapılmaz (AHEPA A1–A2).
  - 1 Hz filtre delta bandının 0.5–1 Hz kısmını bastırır. Bu, önceden bilinen bir protokol farkıdır (§7 duyarlılık).

## 5. Ön işleme

AHEPA ile aynı adımlar uygulanır (`evlib.encode_continuous`):
1. 500 Hz'ten 200 Hz'e MNE `resample(npad="auto")` ile.
2. Sürekli kayıtta 4. derece Butterworth bant filtresi (`sosfiltfilt`).
3. Pencere içi `log10(mean(x_b²) + 1e-12)`.
4. 10 s / %50 pencereleme.

Kayıt kenarı etkisine karşı **kaydın ilk ve son 5 s'si atılır**. AHEPA'da bu yapılmadı; etkisi yalnızca kenar pencerelerle sınırlıdır.

**Pencere kalite kontrolü (hedef veri):**
- AHEPA kayıtları yayıncı tarafından ASR + ICA ile temizlenmişti; Dryad'da artefakt temizliği bildirilmemiş.
- Birincil analizde yalnızca şu bariz kayıt hataları reddedilir:
  - herhangi bir kanalda pencere içi std < 0.1 µV (düz kanal);
  - |x| > 500 µV;
  - NaN.
- Duyarlılık analizinde hiçbir pencere reddedilmez.
- Göz hareketi ya da kas artefaktı için ek temizlik yapılmaz. Bu fark sınırlılık olarak raporlanır ve doygunluk analizinde izlenir.

## 6. Birincil deney (dondurulmuş kaynak kodlayıcı)

1. Her pencere için özgün `lp`, dondurulmuş ölçek kodu ve çözülmüş `lph` hesaplanır.
2. Göreli güç hesaplanır ve katılımcı ortalaması alınır.
3. d~özgün~, d~çözülmüş~ ve kanal-major 95 profil kurulur.
4. Ölçütler: r~d~, Spearman, CCC, bant ortalaması çıkarılmış r~d~, yalnız bant ortalaması profilinin korelasyonu, bant içi r~d~ ve CCC, ortalama |Δd|, bant başına |Δd|, ortalama |d| (özgün ve çözülmüş), top-10 ve top-20 örtüşmesi, log-güç MAE, göreli güç MAE.
5. **Bootstrap:** Grup içi tabakalı katılımcı bootstrap, 2,000 tekrar, tohum 49. Özgün ve çözülmüş profiller aynı örnekte kalır; 95 çift yeniden örneklenmez.
6. **Kanal karıştırma referansı:** 10,000 permütasyon, tohum 42.
7. **Kaynak–hedef özgün profil uyumu:** r(d~AHEPA,özgün~, d~Dryad,özgün~), bant ortalaması çıkarılmış hâli ve bant içi değerler. Ayrı raporlanır. Yüksek r~d~ kaynakla aynı örüntüyü göstermez.

## 7. İkincil ve duyarlılık analizleri

Hepsi açıkça etiketlenir; hiçbiri birincil sonucun yerine geçmez.

| Kod | Analiz | Gerekçe |
|---|---|---|
| S1 | Pencere reddi yok | QC kuralının etkisi |
| S2 | 60 yaş ve üstü kontroller (yaş varsa) | Yaş karıştırıcısı |
| S3 | Delta hariç 4 bant (θ–γ; 19 × 4 = 76). Göreli güç 4 bant üzerinden; AHEPA referansı aynı alt kümeyle yeniden hesaplanır. | 1 Hz yüksek geçiren filtre |
| S4 | **Etiketsiz hedef kalibrasyonu:** PowerQuantizer hedefin bütün katılımcılarından (grup bilgisi kullanılmadan), katılımcı başına 25 rastgele pencereyle (tohum 42) yeniden öğrenilir. AHEPA karşılığı: kendi verisinden öğrenilmiş global nicemleyici (r~d~ 0.978). | Dağılım kaymasının etkisi; tek kalibrasyon düzeyi, sonuca bakılarak seçilmez |
| S5 | Sınıflandırma, seçenek B: Dryad'da katılımcı düzeyinde 5 × 5 CV ile M1 ve M1s, `05_classify.py` ile aynı sınıflandırıcı, ayar ve bölme protokolü. Her grupta en az 20 katılımcı yoksa yapılmaz. **Sıfır atış transfer değildir**, hedefte yeniden eğitimdir. | İkincil |

Seçenek A (AHEPA'da eğitilmiş modelin dondurulmuş transferi) yalnızca Düzey A uyumda yapılır.

## 8. Dağılım kayması tanılamaları

- Bant başına ölçek kodu frekansları.
- En alt (0) ve en üst (7) seviyeye düşen pencere–kanal oranı, yani doygunluk.
- Kod entropisi; azami değer 3 bit.
- Kaynak ve hedef log-güç dağılımları: medyan farkı ve Kolmogorov–Smirnov D.
- Kanal düzeyi güç kayması.
- Grup başına dağılımlar.

**Önceden kaydedilen öngörü:**
- Kesim noktaları dar (0.26–0.67 dekad, `SOURCE_PIPELINE_AUDIT.md` §5).
- ICA ile temizlenmemiş, farklı referanslı ve farklı cihazla alınmış kayıtta önemli doygunluk beklenir; özellikle göz hareketinden etkilenen delta ve kas artefaktından etkilenen gamma bantlarında.
- Doygunluk, çözülmüş göreli gücün varyansını daraltır ve r~d~ ile CCC'yi düşürebilir.
- Bu durum yöntemin "başarısızlığı" olarak değil, sabit mutlak kod defterinin dağılım kaymasına duyarlılığı olarak yorumlanacak ve S4 ile birlikte raporlanacak.
- Etiket kullanılarak hiçbir düzeltme yapılmayacak.

## 9. Karar ölçütleri

**Strong external fidelity support:** Hepsi birlikte sağlanmalı:
- Düzey A uyum;
- r~d~ %95 GA alt sınırı > hedef kanal karıştırma referansının %95 üst değeri;
- CCC ≥ 0.90;
- bant ortalaması çıkarılmış r~d~ ≥ 0.90;
- doygunluk her bantta < %20.

**Partial external fidelity support:** Aşağıdakilerden biri:
- r~d~ GA alt sınırı karıştırma referansının üstünde, ama diğer ölçütlerden en az biri sağlanmıyor;
- ya da protokol Düzey B.

**External fidelity not supported:** Aşağıdakilerden biri:
- r~d~ nokta değeri karıştırma referansının %95 üst değerinin altında;
- ya da CCC < 0.70.

**External validation inconclusive:** Aşağıdakilerden biri:
- GA, "destek" ile "desteklenmiyor" sınırlarını birlikte kapsıyor;
- ya da birimi veya kanal sırası belgelenmemiş bir veri, ölçütleri tek başına belirliyor.

**No compatible independently accessible dataset:** Hiçbir aday Düzey A veya B değil ya da veri elde edilemedi.

Yüksek r~d~ tek başına güçlü doğrulama sayılmaz. Karar CCC, bant içi uyum, |Δd|, doygunluk ve protokol uyumuyla birlikte verilir.

## 10. Yasaklar

- Kesim noktaları, W, a, L, bantlar ve pencere hedef veriye bakılarak değiştirilmez.
- Katılımcı fidelity değerine göre seçilmez.
- Hedef etiketleri hiçbir kalibrasyonda kullanılmaz.
- Ham klinik veri yeniden dağıtılmaz. `02_DATA` paylaşılacak pakete konmaz; yalnızca özetler ve sağlama toplamları paylaşılır.

## 11. Veri görüldükten sonra yapılan değişiklikler

Özgün sürüm `EXTERNAL_PROTOCOL_v1_preregistered.md` olarak değiştirilmeden saklandı (sha256 `11f2b9e7…ef88`).

### 11.1 (2026-11-01) Dosya yapısı görüldükten sonra, hiçbir sadakat ya da sınıflandırma sonucu hesaplanmadan önce

**Ne görüldü** (yalnız yapı ve genlik; `02_DATA/dryad_shor_benninger/metadata/`):
- Üç dosyanın SHA-256 değeri yayıncınınkiyle aynı.
- `alz_r`: 43 katılımcı sütunu, 49 kayıt (bazı katılımcılarda 2–3 kayıt). Alanlar: sinyal `G` (örnek × 19), `age`, `sex`, `birth`, `eegdate`, `eeg_files`, `g` (örnekleme hızı), `m` (belgelenmemiş; 3–30 arası tam sayı, MMSE olabilir), `n` (dosya adı). Yayındaki AH sayısı 40; dosyada 43. MCI işareti yok.
- `controls_r`: 96 katılımcı (yayında 95, README'de 96), alanlar `G`, `age`, `sex`, `g`, `n`. Kontrollerin bir bölümü 200 Hz, diğerleri 500 Hz (ilk yazımda "30" denmişti; manifestoya göre doğru sayı 26/96, düzeltme 2026-11-01); AH'de ilk kayıtların hepsi 500 Hz.
- **Hiçbir dosyada, README'de ya da yayında kanal adı veya kanal sırası yok.**
- Yayın birimi "[mV]" diyor; kanal std değerleri 3–20 aralığında, yani fizyolojik olarak µV ile uyumlu.
- Kayıt süreleri 830–2,725 s (yayında analiz edilen 351 s'lik bölüm dosyada işaretli değil). Kayıt başında sıfır dolgu var.
- Göz durumu işareti yok.

**Uygunluk kararı (§4 uygulanarak):** Kanal sırası belgelenmediği için kanal kimliği gerektiren her analiz yapılamaz ve veri seti protokol §4'e göre kanal-bant profili bakımından **Düzey C**'dir. Bu karar değiştirilmiyor.

**Kanal kimliğinden bağımsız analiz (ek, açıkça etiketli):**
- Dondurulmuş ölçek kodlayıcı yalnız bant başına çalışır (`FrozenScale`, 5 × 7 kesim noktası); kanal kimliği kullanmaz.
- r~d~, Spearman, CCC, bant ortalaması çıkarılmış r~d~, bant içi r~d~/CCC, |Δd|, top-k örtüşmesi ve bant içi kanal karıştırma referansı, kanalların bütün katılımcılarda **aynı (bilinmeyen) sırada** olması koşuluyla, kanal etiketlerinin herhangi bir ortak permütasyonuna karşı değişmezdir.
- Bu nedenle bu ölçütler, kanal adı atanmadan, sütun indeksleri (ch01–ch19) ile hesaplanır. Kanal adı tahmin edilmez, AHEPA kanallarına eşlenmez.
- "Bütün katılımcılarda aynı sıra" varsayımı etiketsiz olarak denetlenir: katılımcı başına 19 kanalın log-std vektörü ve kanallar arası korelasyon matrisi katılımcılar arasında karşılaştırılır (grup bilgisi kullanılmaz). Bu denetim varsayımı kanıtlamaz, yalnızca bariz tutarsızlığı yakalar.
- Kanal kimliği gerektiren analizler (kaynak–hedef kanal-bant profil uyumu, kanal düzeyi güç kayması, AHEPA'da eğitilmiş modelin transferi) **yapılmaz**. Yalnız kanallar üzerinden ortalanmış bant düzeyi uyum (5 değer) raporlanır.

**Önceden sabitlenen karar kuralı (r~d~ görülmeden):**
- §9'daki "birimi veya kanal sırası belgelenmemiş bir veri ölçütleri tek başına belirliyor" maddesi bu durumu tam olarak kapsar.
- Bu yüzden nihai karar, sayısal sonuç ne olursa olsun, **en fazla "External validation inconclusive"** olabilir. Sayılar yalnızca bu kararın içinde, destekleyici/engelleyici bulgu olarak yorumlanır; "Strong" ya da "Partial support" verilmez.
- Sayısal ölçütler yine §9 eşikleriyle karşılaştırılıp raporlanır.

**Birim:** Yayındaki "[mV]" ifadesi, genlikle (kanal std 3–20) çelişiyor; mV kabul edilirse sinyal 3–20 mV olur, fizyolojik değildir. Bu nedenle birim **belgelenmemiş** sayılır ve §4 kuralı uygulanır: geniş bant RMS'nin katılımcılar arası medyanı 5–150 aralığındaysa µV kabul edilir; değilse "çözülemedi" diye raporlanır ve ölçek kodu sonuçları yorumlanmaz. Ölçekleme r~d~'ye bakılarak seçilmez.

**Kayıt seçimi:** Birden fazla kaydı olan AH katılımcılarında yalnız **ilk kayıt** (`G[0, sütun]`) kullanılır. Katılımcı başına tek profil.

**Sıfır dolgu:** Kaydın başındaki ve sonundaki, 19 kanalın hepsinin tam sıfır olduğu örnekler kesilir; ardından §5'teki 5 s kenar kırpması uygulanır. Kayıt içindeki düz bölümler §5 QC'si (std < 0.1 µV) ile reddedilir.

**Örnekleme hızı:** 200 Hz kayıtlar yeniden örneklenmez (hedef zaten 200 Hz). 500 Hz kayıtlar §5'teki gibi 200 Hz'e indirilir.

**Ek duyarlılık analizi S6:** Yalnız 500 Hz kayıtlar (örnekleme hızı/cihaz yapılandırması ile grup kısmen karışık: 200 Hz kayıtların hepsi kontrol).

**S2 yaş sınırı:** Dosyada yaş var; S2 ≥ 60 yaş kontrollerle yapılır (değişmedi).

**S5:** `05_classify.py`'nin özellikleri kanal adı değil sabit sütun sırası kullanıyorsa, aynı varsayım altında yapılır ve aynı etiketle raporlanır.

