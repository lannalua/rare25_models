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
