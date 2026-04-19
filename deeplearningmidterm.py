import os
import shutil
import random
import kagglehub
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# ---------------------------------------------------------
# 1. DATASET İNDİR
# ---------------------------------------------------------
path = kagglehub.dataset_download("omkargurav/face-mask-dataset")
original_dir = os.path.join(path, "data")

# ---------------------------------------------------------
# 2. DATASET SPLIT
# ---------------------------------------------------------
base_dir = "dataset_split"

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

classes = os.listdir(original_dir)

for cls in classes:
    cls_path = os.path.join(original_dir, cls)
    
    if not os.path.isdir(cls_path):
        continue
    
    images = [img for img in os.listdir(cls_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    total = len(images)

    test_split = int(0.2 * total)
    val_split = int(0.2 * (total - test_split))

    test_imgs = images[:test_split]
    val_imgs = images[test_split:test_split+val_split]
    train_imgs = images[test_split+val_split:]

    for img in train_imgs:
        dst = os.path.join(train_dir, cls)
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(os.path.join(cls_path, img), dst)

    for img in val_imgs:
        dst = os.path.join(val_dir, cls)
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(os.path.join(cls_path, img), dst)

    for img in test_imgs:
        dst = os.path.join(test_dir, cls)
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(os.path.join(cls_path, img), dst)

print("Dataset split tamam.")

# ---------------------------------------------------------
# 3. DATA GENERATOR
# ---------------------------------------------------------
image_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ---------------------------------------------------------
# 4. MODEL
# ---------------------------------------------------------
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# ---------------------------------------------------------
# 5. COMPILE
# ---------------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------------------------------------------------
# 6. CALLBACKS
# ---------------------------------------------------------
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    verbose=1
)


class LivePlot(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

        plt.clf()

        plt.subplot(1,2,1)
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.val_loss, label='Val Loss')
        plt.legend()
        plt.title("Loss")

        plt.subplot(1,2,2)
        plt.plot(self.train_acc, label='Train Acc')
        plt.plot(self.val_acc, label='Val Acc')
        plt.legend()
        plt.title("Accuracy")

        plt.pause(0.1)

live_plot = LivePlot()

# ---------------------------------------------------------
# 7. TRAIN
# ---------------------------------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[early_stopping, reduce_lr, live_plot]
)

# ---------------------------------------------------------
# 8. TEST
# ---------------------------------------------------------
print("\n--- TEST SONUÇLARI ---")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

# ---------------------------------------------------------
# 9. TAHMİN
# ---------------------------------------------------------
test_generator.reset()
y_pred_probs = model.predict(test_generator)
y_pred_classes = y_pred_probs.argmax(axis=1)
y_true = test_generator.classes

# ---------------------------------------------------------
# 10. METRİKLER VE TXT OLARAK KAYDETME
# ---------------------------------------------------------
cm = confusion_matrix(y_true, y_pred_classes)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes)
recall = recall_score(y_true, y_pred_classes)
specificity = tn / (tn + fp)
f1 = f1_score(y_true, y_pred_classes)

# Metrikleri ekrana yazdır
print("\n--- METRİKLER ---")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1 Score    : {f1:.4f}")

# YENİ: Metrikleri bir .txt dosyasına kalıcı olarak kaydet
with open("test_metrikleri.txt", "w", encoding="utf-8") as f:
    f.write("--- TEST SETİ METRİKLERİ ---\n")
    f.write(f"Accuracy    : {accuracy:.4f}\n")
    f.write(f"Precision   : {precision:.4f}\n")
    f.write(f"Recall      : {recall:.4f}\n")
    f.write(f"Specificity : {specificity:.4f}\n")
    f.write(f"F1 Score    : {f1:.4f}\n")
print("\n[+] Metrikler 'test_metrikleri.txt' olarak kaydedildi.")

# ---------------------------------------------------------
# 11. CONFUSION MATRIX (PNG OLARAK KAYDETME)
# ---------------------------------------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# YENİ: Grafiği ekranda göstermeden saniyeler önce fotoğraf olarak kaydet
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
print("[+] Confusion Matrix grafiği 'confusion_matrix.png' olarak kaydedildi.")

plt.show()

# ---------------------------------------------------------
# 12. ROC CURVE (PNG OLARAK KAYDETME)
# ---------------------------------------------------------
fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Curve")

# YENİ: Grafiği ekranda göstermeden saniyeler önce fotoğraf olarak kaydet
plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')
print("[+] ROC Eğrisi grafiği 'roc_curve.png' olarak kaydedildi.")

plt.show()

# ---------------------------------------------------------
# 13. MODEL KAYDET
# ---------------------------------------------------------
model.save("best_model.h5")
print("\nModel kaydedildi.")