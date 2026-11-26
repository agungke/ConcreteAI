# ConcreteAI — Presentation Q&A Preparation

Gunakan jawaban singkat berikut sebagai bahan latihan. Sesuaikan jawaban dengan pembagian kerja dan pemahaman aktual setiap anggota.

## 1. Mengapa memilih topik kuat tekan beton?

Kuat tekan merupakan salah satu parameter penting pada beton. Pengujian konvensional dilakukan di laboratorium dan membutuhkan spesimen serta waktu curing. Proyek ini mengeksplorasi apakah data komposisi material dan umur beton dapat digunakan untuk memperkirakan kuat tekan melalui Machine Learning. Sistem ini hanya prototipe akademik.

## 2. Mengapa menggunakan dataset UCI Yeh?

Dataset UCI memiliki 1.030 observasi, delapan fitur numerik dan satu target kuat tekan dalam MPa. Dataset menyediakan komposisi tujuh material dan umur beton, sehingga sesuai dengan rumusan masalah proyek. Dataset berasal dari UCI Machine Learning Repository dan berlisensi CC BY 4.0.

## 3. Mengapa Linear Regression dijadikan baseline?

Linear Regression sederhana, cepat dan mudah dijelaskan secara matematis. Model ini menjadi titik pembanding untuk mengetahui apakah model nonlinear memberikan peningkatan performa.

## 4. Mengapa memilih Random Forest?

Hubungan antara komposisi material, umur curing dan kuat tekan dapat bersifat nonlinear serta melibatkan interaksi antarfitur. Random Forest menggabungkan banyak Decision Tree dan merata-ratakan prediksi, sehingga dapat menangkap pola yang tidak mudah direpresentasikan oleh satu persamaan linear.

## 5. Mengapa tidak menggunakan random train-test split biasa?

Dataset memiliki beberapa observasi dari komposisi campuran yang sama pada umur curing berbeda. Jika split dilakukan secara acak per baris, satu komposisi bisa muncul pada train dan test. Model dapat terlihat lebih baik karena sudah melihat komposisi tersebut. GroupShuffleSplit menggunakan tujuh komponen material sebagai group key dan mengecualikan umur, sehingga satu komposisi tetap berada pada satu sisi split.

## 6. Mengapa hasil split bukan tepat 80:20?

GroupShuffleSplit mempertahankan seluruh group agar tidak terpecah. Karena ukuran tiap group berbeda, proporsi baris aktual menjadi 815 training dan 215 testing (sekitar 79.1:20.9), bukan tepat 80:20. Ini merupakan trade-off yang disengaja untuk mengurangi data leakage.

## 7. Mengapa dilakukan GroupKFold?

GroupKFold menjaga group campuran tetap utuh pada setiap fold. Lima-fold cross-validation dilakukan pada training data saja agar performa model dapat diperiksa pada beberapa pembagian group tanpa menggunakan test set untuk pemilihan model.

## 8. Mengapa tidak melakukan feature scaling?

Linear Regression dengan Ordinary Least Squares dan Random Forest tidak mensyaratkan standardisasi fitur untuk proses fitting seperti model berbasis jarak atau regularisasi tertentu. Fitur dipertahankan pada satuan asli agar interpretasi koefisien tetap terkait dengan kg/m³ dan hari. Standardisasi dapat dipertimbangkan jika metode lain ditambahkan.

## 9. Bagaimana membaca MAE, RMSE dan R²?

- MAE menunjukkan rata-rata besar kesalahan absolut dalam MPa.
- RMSE juga dalam MPa, tetapi memberi penalti lebih besar pada error besar.
- R² membandingkan performa model dengan baseline yang memprediksi rata-rata target. Nilai lebih tinggi menunjukkan variasi target lebih banyak dijelaskan oleh model pada data evaluasi.

Ketiganya perlu dibaca bersama; R² yang tinggi tidak berarti semua prediksi akurat.

## 10. Mengapa Random Forest lebih baik pada hasil ini?

Pada held-out test, Random Forest menghasilkan MAE 4.0332 MPa, RMSE 5.8886 MPa dan R² 0.8933. Linear Regression menghasilkan MAE 8.3832 MPa, RMSE 10.7361 MPa dan R² 0.6453. Group cross-validation juga menunjukkan ranking yang sama. Hasil ini menunjukkan Random Forest lebih baik menangkap pola dataset pada eksperimen ini.

## 11. Apakah Random Forest mengalami overfitting?

Ada indikasi overfitting. Random Forest memiliki train MAE 1.2294 MPa, sedangkan test MAE 4.0332 MPa. R² training 0.9855 turun menjadi 0.8933 pada test. Namun, hasil GroupKFold masih lebih baik dibandingkan Linear Regression. Jadi, model lebih baik dari baseline tetapi belum sempurna dalam generalisasi.

## 12. Mengapa umur beton dan cement menjadi fitur paling penting?

Berdasarkan impurity-based feature importance, age memiliki nilai 34.36% dan cement 28.59%. Ini menunjukkan bahwa Random Forest paling banyak menggunakan kedua fitur tersebut untuk mengurangi impurity pada split pohon. Nilai ini bukan bukti kausal bahwa perubahan satu fitur saja akan menghasilkan perubahan kuat tekan tertentu.

## 13. Mengapa duplicate rows tidak dihapus?

Dataset tidak menyediakan identitas unik untuk setiap spesimen atau pengujian. Baris yang identik dapat berasal dari duplikasi pencatatan atau pengamatan berulang. Karena asal setiap baris tidak dapat dipastikan hanya dari nilai tabel, proyek mempertahankan data dan melaporkan jumlah duplikasi. Group split juga mencegah komposisi identik tersebar ke train dan test.

## 14. Apa arti residual negatif rata-rata?

Residual didefinisikan sebagai aktual dikurangi prediksi. Mean residual -0.43 MPa berarti model sedikit cenderung memprediksi lebih tinggi secara rata-rata pada test set. Ini tidak berarti semua prediksi overestimate; ada 90 underpredictions dan 125 overpredictions.

## 15. Apa arti error maksimum 31.97 MPa?

Ada setidaknya satu sampel dengan selisih absolut prediksi yang sangat besar. Pada salah satu kasus, nilai aktual 81.75 MPa sedangkan prediksi Random Forest 49.78 MPa. Ini menegaskan bahwa performa rata-rata tidak menjamin akurasi untuk setiap campuran, khususnya pada kasus kuat tekan tinggi.

## 16. Apakah aplikasi ConcreteAI bisa menggantikan uji laboratorium?

Tidak. Aplikasi menggunakan data historis yang terbatas dan tidak memvalidasi campuran terhadap standar, supplier, material, proses curing atau kondisi lapangan tertentu. Prediksi hanya untuk demonstrasi akademik dan tidak boleh digunakan sebagai dasar tunggal keputusan keselamatan konstruksi.

## 17. Mengapa tidak menggunakan Deep Learning?

Dataset hanya memiliki 1.030 observasi dan masalahnya dapat dipelajari dengan model klasik. Deep Learning akan menambah kompleksitas dan risiko overfitting tanpa alasan kuat untuk tugas ini. Linear Regression dan Random Forest lebih mudah dijelaskan, diuji dan dipresentasikan.

## 18. Apa pengembangan berikutnya?

Validasi pada dataset independen, permutation feature importance, uncertainty analysis, pemeriksaan error per rentang kuat tekan, dan evaluasi terhadap campuran dari supplier atau kondisi curing yang berbeda.

## 19. Apa kontribusi masing-masing anggota?

Jawab berdasarkan pekerjaan yang benar-benar dilakukan dan bukti commit. Gunakan dokumen docs/team_contribution_template.md. Jangan mengklaim pekerjaan yang hanya direncanakan tetapi belum diselesaikan.

## 20. Apa kontribusi utama proyek?

Kontribusi proyek adalah pipeline regresi yang reproducible, pemisahan data berbasis komposisi campuran untuk mengurangi leakage, perbandingan baseline dan nonlinear model, analisis error, serta aplikasi demonstrasi. Proyek tidak mengklaim menciptakan algoritma Machine Learning baru.
