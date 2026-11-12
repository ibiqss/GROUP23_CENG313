# GROUP23_CENG313: Mining Misconceptions in Mathematics

CENG313 Veri Bilimine Giriş dersi projesi. Proje, Kaggle yarışması
[Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)
üzerine kuruludur.

## Grup 23

| Ad Soyad | Öğrenci No | GitHub | Ödev 1'deki sorumluluk |
|---|---|---|---|
| İbrahim BETBAŞI | 23118080058 | [ibiqss](https://github.com/ibiqss) | Veri yükleme, ön işleme, keşifsel veri analizi, popülerlik baseline'ı |
| Koray CİHAN | 23118080043 | [KorayCihan](https://github.com/KorayCihan) | Doğrulama ve metrik, metin tabanlı baseline'lar, Kaggle gönderimi |

## Yarışma

Her çoktan seçmeli matematik sorusunun bir doğru ve üç yanlış şıkkı vardır. Amaç, her
yanlış şık için öğrencinin bu şıkkı seçmesine yol açmış olabilecek kavram yanılgısını
(misconception) tahmin etmektir. Her yanlış şık için 2.587 aday arasından en olası 25
tanesi sıralanır ve sonuç MAP@25 ile değerlendirilir.

## Veri seti

Veri yarışma tarafından sağlanır ve Kaggle kuralları gereği bu repoya eklenmez
(`data/` klasörü `.gitignore` içindedir).

| Dosya | İçerik |
|---|---|
| `train.csv` | Soru başına bir satır: soru metni, konu, construct, dört şık metni, doğru şık ve her yanlış şık için bir kavram yanılgısı kimliği |
| `test.csv` | Etiketsiz aynı sütunlar (açık dosyada yalnızca 3 soru) |
| `misconception_mapping.csv` | Aday listesi: `MisconceptionId` ve `MisconceptionName` |
| `sample_submission.csv` | Beklenen gönderim formatı |

`src/data.py` soruları yanlış şık başına bir satıra çevirir. Modellerin kullandığı tablo:

| Sütun | Anlamı |
|---|---|
| `QuestionId` | Soru kimliği; doğrulamada gruplama için de kullanılır |
| `Answer` | Yanlış şıkkın harfi (A-D) |
| `QueryText` | Soru, doğru cevap ve yanlış cevabın birleştirilmiş metni |
| `MisconceptionId` | Hedef etiket; etiketsiz satırlar varsayılan olarak çıkarılır |

### Veriyi indirme

Önce Kaggle'da yarışma sayfasından kuralları kabul edin. Sonra veriyi iki yoldan biriyle alın:

- Yarışmanın **Data** sekmesinden **Download All** ile indirip CSV dosyalarını `data/` klasörüne çıkarın.
- Ya da Kaggle API ile indirin:

```bash
kaggle competitions download -c eedi-mining-misconceptions-in-mathematics -p data
```

## Yaklaşım

### Ödev 1: Keşifsel veri analizi ve baseline

Notebook, dersin istediği yedi bölümü izler: problem tanımı, veriye genel bakış, keşifsel
veri analizi, ön işleme, baseline modeller, Kaggle gönderimi ve sonuç. Her grafiğin
ardından yazılı bir yorum gelir.

| Baseline | Fikir |
|---|---|
| Popülerlik | Her satır için eğitimdeki en sık 25 kavram yanılgısını tahmin eder |
| TF-IDF | _Eklenecek_ |
| Cümle gömmesi | _Eklenecek_ |

### Ödev 2: Önerilen çözüm

_Eklenecek._

## Sonuçlar

### Keşifsel veri analizi bulguları

| Bulgu | Değer |
|---|---|
| Soru / yanlış şık | 1.869 / 5.607 |
| Etiketli yanlış şık | 4.370 (%77,9) |
| Aday kavram yanılgısı | 2.587 |
| Train'de hiç geçmeyen aday | 983 (%38,0) |
| Yalnızca bir kez geçen aday | 747 |
| En sık 25 kavram yanılgısının etiket payı | %13,6 |
| Ayrılan sorularda kalan eğitim verisinde olmayan etiketler | ortalama %26,2 |
| LaTeX / görsel içeren sorular | %79,5 / %33,8 |
| Ortalama kelime örtüşmesi, gerçek ve karıştırılmış çiftler | 0,081 ve 0,039 |

Etiketler uzun kuyruklu ve adayların önemli bir kısmı eğitimde hiç geçmiyor. Bu yüzden görev,
sıradan bir sınıflandırmadan çok sorgu metnini kavram yanılgısı metniyle eşleştirme problemi
olarak ele alınmalı. Görseller yalnızca alt metin olarak bulunduğundan bazı yanlış şıklar
metinden ayırt edilemiyor.

### Baseline skorları

| Model | Doğrulama MAP@25 | Kaggle skoru (public / private) |
|---|---|---|
| Popülerlik | _Eklenecek_ | _Eklenecek_ |
| TF-IDF | _Eklenecek_ | _Eklenecek_ |
| Cümle gömmesi | _Eklenecek_ | _Eklenecek_ |

**Güncel Kaggle skoru:** _Eklenecek_

## Çalıştırma

### Google Colab

1. Kaggle'da yarışma kurallarını kabul edin.
2. Colab'da **Secrets** paneline `KAGGLE_USERNAME` ve `KAGGLE_KEY` ekleyin.
3. `assignment1_eda_baseline.ipynb` dosyasını Colab'da açıp tüm hücreleri çalıştırın. Notebook repoyu klonlar, paketleri kurar ve veriyi indirir.

### Yerel

```bash
git clone https://github.com/ceng313-2026/GROUP23_CENG313.git
cd GROUP23_CENG313
pip install -r requirements.txt jupyter
```

Veriyi `data/` klasörüne koyduktan sonra `assignment1_eda_baseline.ipynb` dosyasını açıp tüm hücreleri çalıştırın.

## Repo yapısı

```text
GROUP23_CENG313/
├── README.md
├── requirements.txt
├── .gitignore
├── assignment1_eda_baseline.ipynb   Ödev 1: keşifsel veri analizi ve baseline
├── assignment2_solution.ipynb       Ödev 2: önerilen çözüm (eklenecek)
├── src/
│   ├── __init__.py
│   ├── data.py                      Veri yükleme ve ön işleme
│   ├── models.py                    Baseline modeller
│   └── utils.py                     Veri kalitesi kontrolleri, istatistikler ve grafikler
├── data/                            Commit edilmez
└── submissions/                     Kaggle skor ekran görüntüleri
```

## İlerleme kaydı

| Tarih | Üye | Çalışma |
|---|---|---|
| 2026-11-09 | İbrahim BETBAŞI | Veri yükleme, ön işleme, keşifsel veri analizi ve popülerlik baseline'ı |
| 2026-11-12 | İbrahim BETBAŞI | Notebook dersin yedi bölümlük yapısına göre düzenlendi; her grafiğe yorum eklendi |

## Kaggle gönderim geçmişi

| Tarih | Üye | Model | Public skor | Private skor | Not |
|---|---|---|---|---|---|
| _Eklenecek_ | | | | | |
