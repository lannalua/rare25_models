# ============================================================
# DENSENET-121 — build function com transfer learning
# ============================================================
from tensorflow.keras.applications.densenet import preprocess_input as dn_preprocess #type:ignore  
from main import *
import tensorflow as tf
from tensorflow.keras import layers
def build_densenet121(seed: int = SEED):
    """
    DenseNet-121 com transfer learning em duas fases.
    Fase 1: base congelada — treina só o classificador.
    Fase 2: descongela últimas camadas — fine-tuning com LR baixo.
    """
    tf.random.set_seed(seed)

    base_model = tf.keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # congela para fase 1

    inputs = tf.keras.Input(shape=(224, 224, 3))

    # Pré-processamento esperado pela DenseNet do Keras
    x = tf.keras.applications.densenet.preprocess_input(inputs)
    x = base_model(x, training=False)  # training=False mantém BN congelado
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        256, activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
    )(x)
    x = tf.keras.layers.Dropout(0.5, seed=seed)(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+1)
    )(x)

    model = tf.keras.Model(inputs, outputs, name="densenet121_fase1")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),  # LR maior na fase 1
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )
    return model, base_model


def unfreeze_and_compile(model, base_model, unfreeze_from: int = -50, seed: int = SEED):
    """
    Fase 2: descongela as últimas `unfreeze_from` camadas da base
    e recompila com learning rate muito menor para fine-tuning.
    """
    base_model.trainable = True

    # Congela tudo exceto as últimas camadas
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    # BN das camadas congeladas deve continuar em modo inferência
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # LR muito menor no fine-tuning
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )
    return model


# ============================================================
# AUGMENTATION SELETIVO — apenas classe positiva (label = 1)
# ============================================================

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Augmentation já definido anteriormente (sem RandomBrightness)
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.2),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1),
], name="augmentation")


def augment_positive_only(image, label):
    """
    Aplica augmentation apenas se label == 1.
    tf.cond é necessário porque o grafo do TF não aceita if Python normal
    sobre tensores — o valor de label só é conhecido em runtime.
    """
    image = tf.cond(
        tf.equal(label, 1),
        true_fn=lambda: data_augmentation(image, training=True),
        false_fn=lambda: image
    )
    return image, label


def build_dataset(X, y, shuffle=False, augment=False):
    """
    Constrói um tf.data.Dataset a partir de arrays NumPy.

    Parâmetros
    ----------
    X       : array (N, H, W, 3)
    y       : array (N,)
    shuffle : embaralha a cada época — usar só no treino
    augment : ativa augmentation seletivo — usar só no treino
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        # buffer_size = len(X) garante embaralhamento completo
        ds = ds.shuffle(buffer_size=len(X), seed=SEED,
                        reshuffle_each_iteration=True)

    if augment:
        ds = ds.map(augment_positive_only, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


val_ds = build_dataset(X_val,  y_val,  shuffle=False, augment=False)
test_ds = build_dataset(X_test, y_test, shuffle=False, augment=False)

# ============================================================
# TREINO 1 — DenseNet pura (class_weight)
# ============================================================
set_global_seed(SEED)
model_dn, base_dn = build_densenet121(SEED)

# --- Fase 1: classificador ---
early_stop_dn_f1 = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

print("=== DenseNet-121 | Fase 1 — classificador ===")
start = time.time()

history_dn_f1 = model_dn.fit(
    build_dataset(X_train, y_train, shuffle=True, augment=False),
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weight,
    callbacks=[early_stop_dn_f1],
    verbose=1,
)

# --- Fase 2: fine-tuning ---
model_dn = unfreeze_and_compile(
    model_dn, base_dn, unfreeze_from=-20, seed=SEED)

early_stop_dn_f2 = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

print("\n=== DenseNet-121 | Fase 2 — fine-tuning ===")

history_dn_f2 = model_dn.fit(
    build_dataset(X_train, y_train, shuffle=True, augment=False),
    validation_data=val_ds,
    epochs=20,                   # mais épocas — LR baixo converge devagar
    class_weight=class_weight,
    callbacks=[early_stop_dn_f2],
    verbose=1,
)

total_dn = time.time() - start
print(f"\nTempo total DenseNet S1: {total_dn:.2f}s")

# --- Avaliação ---
y_probs_dn_s1 = model_dn.predict(test_ds).ravel()
print("\n" + "=" * 45)
print("DenseNet-121 S1 — class weight")
print("=" * 45)
full_evaluation_report(y_test, y_probs_dn_s1, threshold=0.5, seed=SEED)

# ============================================================
# DATASETS
# ============================================================
train_ds = build_dataset(X_train, y_train, shuffle=True,  augment=True)
val_ds = build_dataset(X_val,   y_val,   shuffle=False, augment=False)
test_ds = build_dataset(X_test,  y_test,  shuffle=False, augment=False)

class BalancedBatchGenerator(Sequence):
    def __init__(self, X, y, batch_size=32, seed=SEED, preprocess_fn=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.seed = seed
        self.n_per_class = batch_size // 2
        self._epoch = 0
        self.preprocess_fn = preprocess_fn   # <-- novo parâmetro

        self.indices_pos = np.where(y == 1)[0]
        self.indices_neg = np.where(y == 0)[0]
        self._n_steps = int(np.ceil(len(self.indices_pos) / self.n_per_class))
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, index):
        step_seed = self.seed + self._epoch * 10000 + index
        rng = np.random.default_rng(step_seed)

        idx_pos = rng.choice(self.indices_pos, self.n_per_class, replace=True)
        idx_neg = rng.choice(self.indices_neg, self.n_per_class, replace=True)

        batch_indices = np.concatenate([idx_pos, idx_neg])
        rng.shuffle(batch_indices)

        X_batch = self.X[batch_indices]

        # Desfaz o /255.0 e aplica o preprocess correto da arquitetura
        if self.preprocess_fn is not None:
            X_batch = self.preprocess_fn(X_batch * 255.0)

        return X_batch, self.y[batch_indices]

    def __len__(self):
        return self._n_steps

    def on_epoch_end(self):
        self._epoch += 1
# ============================================================
# INTEGRAÇÃO COM O AUGMENTATION SELETIVO
# ============================================================
# O generator retorna arrays NumPy por batch — para aplicar
# o augmentation seletivo, convertemos para tf.data via
# from_generator depois de instanciar.


train_generator = BalancedBatchGenerator(
    X_train, y_train, batch_size=BATCH_SIZE, seed=SEED, preprocess_fn=dn_preprocess
)

# Converte para tf.data mantendo o augmentation seletivo
train_ds_balanced = tf.data.Dataset.from_generator(
    generator=lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,),              dtype=tf.int64),
    )
).map(
    # augment_positive_only opera imagem a imagem — unbatch/map/batch
    lambda X_batch, y_batch: tf.map_fn(
        lambda pair: augment_positive_only(pair[0], pair[1]),
        (X_batch, y_batch),
        fn_output_signature=(tf.float32, tf.int64)
    ),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)
