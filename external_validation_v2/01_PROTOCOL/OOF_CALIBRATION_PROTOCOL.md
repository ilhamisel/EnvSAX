# External Validation V2 – Katılımcı-ayrık, etiketsiz ölçek kodu kalibrasyonu: protokol

Yazım tarihi: 2026-11-02. Bu protokol, V2 deneyinin hiçbir sonucu hesaplanmadan önce yazıldı ve `PROTOCOL_LOCK.json` ile sabitlendi (sha256).

**Konum.** Bu deney ilk dış değerlendirmeden (EnvSAX_EXTERNAL_VALIDATION, 2026-11-01) **sonra** tasarlanmıştır. O değerlendirmenin bütün sonuçları (dondurulmuş kod r_d = 0.804; tüm kohortta etiketsiz yeniden uyum r_d = 0.992) bilinmektedir. V2 önceden kayıtlı bir dış doğrulama değil, dış değerlendirmeyi izleyen **yöntemsel güçlendirme deneyidir**.

## 1. Soru

Hedef kohortta (Dryad, Benninger–Shor) sınıf etiketi kullanılmadan öğrenilen bir EnvSAX ölçek kod defteri, kod defterinin öğrenilmesinde kullanılmamış katılımcılarda AH–kontrol spektral etki büyüklüğü profilini koruyor mu?

## 2. Değişmeyenler

- W = 40, a = 8, L = 8; bantlar, 10 s / %50 pencere, filtre, Hilbert zarfı: `experiments/config.py` (değiştirilmez).
- Ön işleme ve pencere QC: ilk dış deneydeki `02_prepare_dryad.py` çıktıları (`02_DATA/dryad_shor_benninger/derived/lp_<id>.npz`) olduğu gibi kullanılır; yeniden ön işleme yapılmaz.
- Dahil etme: ilk deneydeki kural (≥ 6 temiz pencere) → 43 AH, 96 kontrol (139). Kimse sonradan dışlanmaz.
- Nicemleyici tanımı: `symbolic.PowerQuantizer` ile aynı (bant başına log10 güç üzerinde 7 ampirik kuantil kesim noktası; temsil değeri = seviye içindeki kalibrasyon değerlerinin ortalaması).
- Ölçütler: `evlib.fidelity` ve `U.cohens_d` (ilk deneyle aynı tanımlar). Profil: katılımcı-ortalama doğrusal göreli bant gücü, 19 sütun × 5 bant = 95.

## 3. Katılımcı bölmesi (birincil)

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`; sıra `participant_manifest_analysis.csv` sırası; tabakalama değişkeni yalnızca grup etiketi.
- Etiket yalnızca bölmeyi kurmak için kullanılır. Kalibrasyon fonksiyonu etiket argümanı almaz.
- Her katılımcı tam olarak bir kez held-out olur. Bölme `PARTICIPANT_FOLDS.csv` olarak sonuçlardan önce yazılır ve kilitlenir.
- Birden fazla kaydı olan AH katılımcılarında yalnızca ilk kayıt kullanıldığından (ilk deney) bir katılımcının farklı kayıtları kalibrasyon ve değerlendirmeye bölünemez.

## 4. Her kat

1. **Kalibrasyon verisi:** yalnızca kalibrasyon katılımcıları; her birinden QC'den geçen pencereler arasından 25 pencere (yerine koymadan). Bu, AHEPA dondurulmuş kodlayıcısı ve ilk deneydeki S4 ile aynı katılımcı başına pencere sayısıdır.
   - Fark (raporlanır): S4'te tek bir RNG manifesto sırasıyla ilerliyordu. V2'de pencere seçimi her katılımcı için `default_rng([42, kat, crc32(id)])` ile yapılır; böylece kod defteri katılımcı sırasından bağımsızdır.
   - AHEPA'daki makale OOF r_d'si (0.974) eğitim katının **bütün** pencereleriyle öğrenilmişti; V2 25 pencereyi kullanır. Bu protokol farkı raporlanır.
2. **Kod defteri:** bant başına 7 kesim noktası + 8 temsil değeri; `CODEBOOK_fold{k}.npz` olarak kalibrasyon katılımcı listesiyle birlikte kaydedilir.
3. **Held-out dönüşüm:** kod defteri değiştirilmeden held-out katılımcıların QC'den geçen bütün pencerelerine uygulanır (ilk deneydeki değerlendirme ile aynı). Held-out veriden hiçbir sınır ya da merkez öğrenilmez.
4. **Kayıt:** katılımcı × kanal × bant düzeyinde özgün/çözülmüş log güç ortalaması, özgün/çözülmüş göreli güç ortalaması, sembol dağılımı.

## 5. Birincil ölçüm

139 katılımcının OOF çözülmüş göreli güç profilleri birleştirilir; AH ve kontrol arasında 95 kanal-bant Cohen's d hesaplanır. Birincil metrik `r_d = corr(d_original, d_OOF)`. Kat içi küçük test gruplarının r_d ortalaması birincil **değildir**; kat bazlı değerler yalnızca betimleyici verilir.

Ek ölçütler (birlikte yorumlanır): Spearman, Lin CCC, bant ortalaması çıkarılmış r_d, bant içi r_d ve CCC, ortalama |Δd|, özgün ve çözülmüş ortalama |d|, top-10/top-20 örtüşme, log-güç MAE, göreli güç MAE, uç seviye doygunluğu, sembol entropisi.

## 6. Belirsizlik

- **Birincil GA:** 2.000 tekrar, grup içi tabakalı katılımcı bootstrap'ı, `seed = 49` (ilk deneydeki `C.SEED + 7` ile aynı; aynı katılımcı yeniden örneklemeleri). OOF özellikleri sabit tutulur; GA, kod defteri yeniden öğrenme değişkenliğini kapsamaz.
- **Eşleşmiş farklar:** aynı bootstrap örnekleri üzerinde C3 − C1 ve C3 − C2.
- **Kanal karıştırma referansı:** bant içinde kanal sırası karıştırılır (10.000 permütasyon, seed 42), ilk deneyle aynı.
- **İkincil (kod defteri öğrenme değişkenliği):** `RepeatedStratifiedKFold(5, n_repeats=10, random_state=2026)` ile 10 farklı bölme; her biri için tam OOF r_d, CCC, bant ortalaması çıkarılmış r_d. Hepsi raporlanır, hiçbiri seçilmez.

## 7. Kod defteri kararlılığı ve doygunluk

Kat başına kesim noktaları ve temsil değerleri; bant × kesim noktası için katlar arası ortalama, SD, aralık; tam kohort yeniden uyumu (C2) ve AHEPA dondurulmuş kod defterinden fark; held-out kodlarda seviye kullanım oranları, uç seviye doygunluğu (seviye 0 + 7), entropi. Karşılaştırma: AHEPA held-out 26 (dondurulmuş), Dryad dondurulmuş.

## 8. Önceden belirlenmiş kontroller

- **Sızıntı:** her katılımcı için özellikleri üreten katın kalibrasyon listesinde o katılımcının bulunmadığı otomatik olarak doğrulanır; kalibrasyon pencerelerinin hepsi kalibrasyon katılımcılarına aittir; kat birleşimi 139 katılımcıyı tam bir kez kapsar.
- **Etiket bağımsızlığı:** kalibrasyon katılımcı sırası rastgele karıştırılıp kod defteri yeniden öğrenildiğinde aynı (bit düzeyinde) olmalıdır; kalibrasyon fonksiyonunun imzasında etiket yoktur.
- **Birim değişmezliği:** sinyalin sabit bir çarpanla ölçeklenmesi log güce sabit ekler; kuantil kod defteri buna eşdeğişkendir, göreli güç değişmez. Log güce +6 (µV ↔ mV, güçte 10^6) eklenerek C3 profilinin değişmediği doğrulanır. Bu kontrol birim **seçmek** için kullanılmaz; birim kuralı ilk deneydeki gibi kalır.
- **C2 yeniden üretimi:** kaydedilmiş S4 kod defteri V2 kodu ile değerlendirildiğinde r_d = 0.99185 elde edilmelidir.

## 9. Önceden belirlenmiş duyarlılık analizi

S-cc: etiketsiz kanal tutarlılığı denetiminde (ilk deney, `channel_consistency.csv`) `r_corr < 0.5` olan katılımcılar çıkarılarak birincil bölmenin OOF özellikleri üzerinde ölçütler yeniden hesaplanır (kod defterleri yeniden öğrenilmez). Bu analiz sonucu değiştirmek için değil, kanal sırası varsayımına duyarlılığı göstermek için yapılır.

## 10. Karar kuralı (sonuçlardan önce)

Eşikler, ilk dış protokoldeki §9 (sonuçlardan önce yazılmış) eşiklerinden ve AHEPA kohort-içi referansından alınmıştır. Doygunluk ölçütü uyarlanmıştır: 8 eş frekanslı seviyede uç iki seviyenin nominal payı %25'tir; ilk protokoldeki "%20'nin altı" ölçütü kuantil kalibrasyonunda tanım gereği sağlanamaz. AHEPA held-out referansı %20–28'dir.

Ölçütler (C3, 139 katılımcı OOF):
1. r_d %95 GA alt sınırı > C3 kanal karıştırma referansının %95 değeri.
2. CCC ≥ 0.90.
3. Bant ortalaması çıkarılmış r_d ≥ 0.90 **ve** GA alt sınırı ≥ 0.70.
4. Ortalama |Δd| ≤ 0.11 (AHEPA 65 katılımcı makale değeri 0.108).
5. Held-out uç seviye doygunluğu her bantta ≤ 0.35.
6. Kod defteri kararlılığı: her bantta her iç kesim noktasının katlar arası aralığı ≤ 0.10 log10 birimi (ilk deneyde kaynak–hedef kayması 0.3–1.8 dekad, kesim aralıkları 0.26–0.67 dekad).

Kategoriler (sırayla değerlendirilir):
- **Not supported:** r_d nokta değeri < karıştırma q95 **ya da** CCC < 0.70.
- **Inconclusive:** bant ortalaması çıkarılmış r_d GA'sı hem < 0.50 hem ≥ 0.90 değerlerini kapsıyor; **ya da** ikincil 10 bölmede bant ortalaması çıkarılmış r_d en küçük değeri < 0.50 ve en büyük değeri ≥ 0.90.
- **Participant-disjoint calibration supported:** 1–6'nın hepsi.
- **Partially supported:** 1 sağlanıyor ve CCC ≥ 0.70, ama 2–6'dan en az biri sağlanmıyor.

Kategori ne olursa olsun: kanal sırası belgelenmediği için kanal düzeyindeki yorum keşifseldir; sonuç dondurulmuş merkezler arası aktarım başarısı, klinik tanı genellenebilirliği ya da metadata sorunlarının çözümü olarak sunulmaz.

## 11. Yasaklar

Yeni veri indirilmez; yeni model eğitilmez; W/a/L, bantlar, pencere sayısı, bölme ve tohumlar sonuçlara göre değiştirilmez; başka bölme denenip en iyisi seçilmez; kat seçilmez; hedef etiketleri kalibrasyonda kullanılmaz; olumsuz sonuç gizlenmez; makale dosyası değiştirilmez; eski deney klasörleri değiştirilmez.
