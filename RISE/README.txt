Merhaba,

Bunun standart bir formatta readme dosyası olmadığını farkındayım ancak süre kısıtlı ve görev hedefe yönelik olduğundan formal olmayan bir yazmayı daha uygun gördüm.

Öncelikle size attığım maile rağmen görevi açıkça anlama konusunda cevap alamadığım için için görev 2 kısımdan oluşmakta. İlki Program_to_MRI_combine klasörüyle ilişkili, bu program datasette verilen vücut parçalarını, eşleştiriyor ve donmadan önce, donmadan sonra ve diğer kondisyon sağlandığı aşamada birleştiriyor. Buradaki image_load kısmı veri setinin yüklemesinin yapıldığı yer, combiner resimleri birleştirmek için, displayer ise matplotlib kullanarak sergilemek için. Ana operasyon da main.py üzerinden ilerliyor. Programın bu kısmı sadece vücut parçalarının birleşmesiyle MR görüntü sentezi yapıyor.

İkinci kısımda hedefim maildeki talimatlarda da belirtildiği şekilde MR görüntüleri kullanarak Ultrason görüntülerini elde etmekti. Anca veri setinde modeli eğitmek için kullanabileceğim bir ultrason görüntüsünü bulamadım. Zaman azaldığından ve hızlı aksiyon almam da gerektiğinden bu kısımda Bilgisayarlı Tomografi verilerini kullanarak renkli gerçek kesitleri elde etmeye çalışıyorum çünkü vücut parçaları burada uyumluluk gösteriyor.

İkinci kısımda algoritma prensibinde:
load_images.py, gerekli CT verilerini yüklememe yarıyor.
preprocessed_images.py kısmında yeniden ölçeklendirme, normalleştirme ve öğrenme aşamasını kolaylaştırmasına yönelik verileri augmente etme yöntemiyle transpose ve rotate işlemlerine sokuyorum.

Bu noktada main kısmına dönelim, main kısmında verilerin dosya yolunu kullanma vasıtasıyla görüntüler arasında dönerek görüntüleri Pıllow formatına getiriyorum ve bunları augmentation işlemine sokuyorum. Bundan sonra görüntüleri eğitim aşamasında kolaylık sağlamasına yönelik numpay arraylerine dönüştürüyorum. Bu arrayler Preprocessed_MRI kısmında tutuluyor.

Sonrasında training_loader kısmına geçiyorum. Burada pytorch kullanarak numpy array formatındaki CT resimlerini ve gerçek resimleri yüklüyorum.

Bundan sonra train kısmı, train kısmında CT görselleri ve Gerçek organ kesitleriyle birleştirme yapmaya çalışıyorum ve bu sırada doğru olmamasına rağmen validation ve test kısımlarını yazıyorum ancak train kısmını hata ayıklama işlemine sokma aşamasında akşam saatlerine geldiğimden sürem bitiyor. Proje ile ilgili kısıtlı yetkinliğim ve zamanın kısıtlı olması gibi faktörlerden dolayı elimden gelen bu, yine de zamanınızı ayırdığınız için teşekkürler.

Saygılarımla