# =======================
# Import library yang dibutuhkan
# =======================
# Semua import Keras sebaiknya lewat tensorflow.keras (bukan keras terpisah)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Scikit-learn untuk preprocessing & evaluasi
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Library tambahan
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# =======================
# Hyperparameter training
# =======================
INIT_LR = 1e-4   # learning rate
EPOCHS = 20      # jumlah epoch
BS = 32          # batch size

# =======================
# Lokasi dataset
# =======================
# Sesuaikan path dataset dengan struktur projectmu
# Misalnya dataset/with_mask dan dataset/without_mask ada di folder project
DIRECTORY = r"dataset"
CATEGORIES = ["with_mask", "without_mask"]

# =======================
# Load dataset
# =======================
print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        # load gambar dengan ukuran 224x224 (standar MobileNetV2)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# =======================
# Encode label ke one-hot
# =======================
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split data train/test
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)

# =======================
# Data augmentation
# =======================
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# =======================
# Load model base MobileNetV2
# =======================
# include_top=False artinya kita tidak pakai layer fully connected default dari MobileNetV2
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# Tambahkan "head" custom untuk klasifikasi mask vs no-mask
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Gabungkan base + head
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze layer base agar tidak ikut dilatih (transfer learning)
for layer in baseModel.layers:
    layer.trainable = False

# =======================
# Compile model
# =======================
print("[INFO] compiling model...")
# Catatan: di TF 2.x argumen 'lr' sudah deprecated → pakai learning_rate
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# =======================
# Training
# =======================
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)

# =======================
# Evaluasi model
# =======================
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# =======================
# Simpan model
# =======================
print("[INFO] saving mask detector model...")
model.save("mask_detector.h5")  # save_format="h5" otomatis

# =======================
# Plot hasil training
# =======================
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
