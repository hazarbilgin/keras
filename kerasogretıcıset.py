import keras
from matplotlib import units
#keras eğitim seti model yapısını 
#classları ve keras functionlarını öğrenme seti

#                       model sınıfı:
keras.Model()

#Katmanları eğitim/çıkarım özelliklerine 
# sahip bir nesnede gruplandıran bir model.
#Bir örneği oluşturmanın üç yolu vardır:Model

#               işlevsel API ile
#'den başlarsınız, modelin ileri geçişini belirtmek için katman çağrılarını zincirlersiniz, Ve son olarak
# modelinizi girdi ve çıktılardan oluşturursunuz:Input

inputs = keras.Input(shape=(37,))
x = keras.layers.Dense(32, activation="relu")(inputs)
outputs = keras.layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
#Kullanılarak yeni bir İşlevsel API modeli de oluşturulabilir. 
# ara tensörler. Bu, alt
# bileşenleri hızlı bir şekilde çıkarmanızı sağlar modelin.

#örnek
inputs = keras.Input(shape=(None, None, 3))
processed = keras.layers.RandomCrop(width=128, height=128)(inputs)
conv = keras.layers.Conv2D(filters=32, kernel_size=3)(processed)
pooling = keras.layers.GlobalAveragePooling2D()(conv)
feature = keras.layers.Dense(10)(pooling)

full_model = keras.Model(inputs, feature)
backbone = keras.Model(processed, conv)
activations = keras.Model(conv, feature)
#Note that the and models are not created with keras.
# Input objects, but 
# with the tensors that originate from keras.Input objects.
# Under the hood, the layers and weights will be shared across 
# these models, so that user can train the , 
# and use or to do feature extraction.
# The inputs and outputs of the model 
# can be nested structures of tensors as well, and the created 
# models are standard Functional API models that support all 
# the existing APIs.backboneactivationsfull_modelbackboneactivations





#Sınıfı alt sınıflara ayırarak Model yapımı
#örnek:
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(5, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = MyModel()
#Alt sınıfa eklerseniz, isteğe bağlı olarak 
# belirtmek için kullanabileceğiniz bir 
# bağımsız değişken (boolean) Eğitim ve çıkarımda
# farklı bir fonksiyon kullanabilirsiniz:Modeltrainingcall()
#örnek:
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(5, activation="softmax")
        self.dropout = keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)

model = MyModel()
#Ek olarak, keras. Sıralı, modelin özel bir durumudur. Model 
# tamamen tek girdili, tek çıkışlı katmanlardan oluşan bir yığındır.
model = keras.Sequential([
    keras.Input(shape=(None, None, 3)),
    keras.layers.Conv2D(filters=32, kernel_size=3),
])

#summary
model.summary(
    line_length=None,
    positions=None,
    print_fn=None,
    expand_nested=False,
    show_trainable=False,
    layer_range=None,
)

#                                                                     Arguments

#line_length: Yazdırılan satırların toplam uzunluğu (örneğin, ekranı farklı bir şekilde uyarlamak için bunu ayarlayın. terminal penceresi boyutları).
#positions: Günlük elemanlarının göreli veya mutlak konumları her satırda. Sağlanmazsa, olur. Varsayılan olarak .[0.3, 0.6, 0.70, 1.]None
#print_fn: Kullanılacak yazdırma işlevi. Varsayılan olarak, öğesine yazdırır. Ortamınızda çalışmıyorsa, olarak değiştirin. Özetin her satırında çağrılacaktır. Bunu özel bir işleve ayarlayabilirsiniz dize özetini yakalamak için.stdoutstdoutprint
#expand_nested: İç içe modellerin genişletilip genişletilmeyeceği. Varsayılan olarak .False
#show_trainable: Bir katmanın eğitilebilir olup olmadığının gösterilip gösterilmeyeceği. Varsayılan olarak .False
#layer_range: 2 dizeden oluşan bir liste veya demet, başlangıç katmanı adı ve bitiş katmanı adıdır (her ikisi de dahil) yazdırılacak katmanların aralığını gösterir Özetle. Ayrıca tam yerine normal ifade kalıplarını da kabul eder ad. Bu durumda, başlangıç yüklemi ilk unsur olacaktır ile eşleşir ve son yüklem şöyle olur: eşleştiği son öğe. Varsayılan olarak, modelin tüm katmanlarını dikkate alır.layer_range[0]layer_range[1]None

#getlayer:
Model.get_layer(name=None, index=None)
#Retrieves a layer based on either its name (unique) or index.
#If and are both provided, will take precedence. Indices are based on order of horizontal graph traversal (bottom-up).nameindexindex

#                                               Arguments

#name: String, name of layer.
#index: Integer, index of layer.

#-------------------------------------**************************---------------------------------------------------------------**************---------
#şimdi birazda dense fonksiyonuna bakalım bu fonksiyo
#katmanlar arası geçişteki düğüm gibidir birinin çıktısını 
#diğerinin girdisi yapabilir şimdi densede kullanılan birimlere
#                       bakalım ve açıklayalım:

# units: Pozitif tamsayı, çıktı uzayının boyutluluğu.
#activation: Kullanılacak aktivasyon fonksiyonu. Hiçbir şey belirtmezseniz etkinleştirme uygulanmaz (yani. "doğrusal" aktivasyon: ).a(x) = x
#use_bias: Boolean, katmanın bir sapma vektörü kullanıp kullanmadığı.
#kernel_initializer: Ağırlık matrisi için başlatıcı.kernel
#bias_initializer: Önyargı vektörü için başlatıcı.
#kernel_regularizer: Regularizer işlevi aşağıdakilere uygulanır: Ağırlıklar matrisi.kernel
#bias_regularizer: Önyargı vektörüne uygulanan düzenlileştirici işlevi.
#activity_regularizer: Regularizer işlevi uygulanır katmanın çıktısı ("aktivasyonu").
#kernel_constraint: Uygulanan kısıtlama işlevi Ağırlıklar matrisi.kernel
#bias_constraint: Sapma vektörüne uygulanan kısıtlama işlevi.
#lora_rank: İsteğe bağlı tamsayı. Ayarlanırsa, katmanın ileri geçişi LoRA'yı (Düşük Dereceli Uyarlama) uygulayacak sağlanan rütbe ile. LoRA, katmanın çekirdeğini ayarlar eğitilemez hale getirir ve üzerinde bir delta ile değiştirir. orijinal çekirdek, iki alt sıranın çarpılmasıyla elde edilir eğitilebilir matrisler. Bu, azaltmak için yararlı olabilir. Büyük yoğun katmanların ince ayarının hesaplama maliyeti. Ayrıca mevcut bir katmanda LoRA'yı arayarak etkinleştirebilirsiniz.Denselayer.enable_lora(rank)

#           örnek DENSE LAYER , DENSE CLASS
keras.layers.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    lora_rank=None,
)

#giriş şekli: 
# Şekilli N-D tensörü: . En yaygın durum şu olacaktır şekilli bir 2D giriş
#(batch_size, ..., input_dim)(batch_size, input_dim)
#ÇIKTI ŞEKLİ:
# Şekilli N-D tensörü: . Örneğin, şekilli bir 2D giriş için, çıktı 
# şekli olacaktır.(batch_size, ..., units)(batch_size, input_dim)(batch_size, units)



#           CORE LAYERS KERAS 3 APU
#Input object
#InputSpec object
#Dense layer
#EinsumDense layer
#Activation layer
#Embedding layer
#Masking layer
#Lambda layer
#Identity layer