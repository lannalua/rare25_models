from tensorflow.keras import layers
import tensorflow as tf
from main import *
# ====MODELO CNN — inicializadores com seed para reprodutibilidade

def build_cnn(seed: int = SEED):
    """Constrói a CNN com inicializadores determinísticos."""
    init_w = GlorotUniform(seed=seed)
    init_b = Zeros()

    model = Sequential([
        # Bloco 1
        Conv2D(32, (3, 3), activation="relu",
               input_shape=(224, 224, 3),
               kernel_initializer=init_w,
               bias_initializer=init_b),
        MaxPooling2D(pool_size=(2, 2)),

        # Bloco 2
        Conv2D(64, (3, 3), activation="relu",
               kernel_initializer=GlorotUniform(seed=seed + 1),
               bias_initializer=init_b),
        MaxPooling2D(pool_size=(2, 2)),

        # Classificador
        Flatten(),
        Dense(128, activation="relu",
              kernel_initializer=GlorotUniform(seed=seed + 2),
              bias_initializer=init_b),
        Dropout(0.5, seed=seed),          # Dropout também recebe seed
        Dense(1, activation="sigmoid",
              kernel_initializer=GlorotUniform(seed=seed + 3),
              bias_initializer=init_b),
    ], name="cnn_v1")

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )
    return model


model = build_cnn(SEED)
model.summary()

# ============================================================
# 7. CALLBACKS
# ============================================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
# ============================================================
# 8. TREINO
# ============================================================
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1,
)

total_time = time.time() - start_time
print(f"\nTempo total de treino: {total_time:.2f}s")

# ============================================================
# 9. AVALIAÇÃO — val e teste
# ============================================================
print("\n=== Validação ===")
val_results = model.evaluate(X_val, y_val, verbose=0)
for name, value in zip(model.metrics_names, val_results):
    print(f"  {name}: {value:.4f}")

print("\n=== Teste ===")
test_results = model.evaluate(X_test, y_test, verbose=0)
for name, value in zip(model.metrics_names, test_results):
    print(f"  {name}: {value:.4f}")

# ===== Métricas 

# shape (N, 1) — .ravel() aplicado internamente
y_probs_cnn = model.predict(X_test)

print("=" * 40)
print("CNN")
print("=" * 40)
full_evaluation_report(y_test, y_probs_cnn, threshold=0.5, seed=SEED)

# ======= CNN + Data Augmentation
# ============================================================
# TREINO 2 — com augmentation seletivo
# RODAR APÓS o treino 1 já ter sido avaliado e salvo
# ============================================================

# 1. Reseta todos os geradores de aleatoriedade
set_global_seed(SEED)

# 2. Reconstrói o modelo do zero — pesos iniciais idênticos ao treino 1
#    Isso garante que a única variável entre os dois treinos é o augmentation
model_aug = build_cnn(SEED)

# 3. Reconstrói o callback — o EarlyStopping guarda estado interno
#    (melhor val_loss, contador de patience) do treino anterior
early_stop_aug = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# 4. Treina com o dataset aumentado
history_aug = model_aug.fit(
    train_ds,                      # dataset com augmentation seletivo
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[early_stop_aug],
    verbose=1,
)

# 5. Avalia
y_probs_aug = model_aug.predict(test_ds).ravel()
full_evaluation_report(y_test, y_probs_aug, threshold=0.5, seed=SEED)
