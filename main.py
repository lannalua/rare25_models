#importação do dataset rare25 via kaggle
from tensorflow.keras.utils import Sequence
import time
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, Zeros
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import random
import os
from sklearn.model_selection import train_test_split
from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers

caminho_do_dataset = "/content/RARE25-train/train"

try:
    dataset = load_from_disk(caminho_do_dataset)
    print("✅ Dataset carregado com sucesso!")
    print(f"Estrutura do dataset: {dataset}")
except Exception as e:
    print(f"❌ Erro ao carregar: {e}")

if 'dataset' in locals():
    print("\nExemplo do primeiro item:")
    primeiro_item = dataset[0]
    segundo_item = dataset[1]
    print(primeiro_item.keys()) 

   
    coluna_imagem = None
    if 'image' in primeiro_item:
        coluna_imagem = 'image'
    elif 'img' in primeiro_item:
        coluna_imagem = 'img'

    if coluna_imagem:
        print(f"\nVisualizando uma amostra da classe: {primeiro_item.get('label', 'Desconhecido')}")
        plt.imshow(primeiro_item[coluna_imagem])
        plt.axis('off')
        plt.title(f"Amostra 0 - Label: {primeiro_item.get('label', 'N/A')}")
        plt.show()
    else:
        print("\nNão encontrei uma coluna com nome óbvio de imagem. Quais são as chaves impressas acima?")

# --------- Contagem das imagens
labels = dataset['label']
total = len(labels)

# Contagem
contagem_0 = labels.count(0)
contagem_1 = labels.count(1)

print(f"Total de imagens: {total}")
print(f"Classe 0 (Negativo): {contagem_0} ({(contagem_0/total)*100:.2f}%)")
print(f"Classe 1 (Positivo): {contagem_1} ({(contagem_1/total)*100:.2f}%)")

plt.bar(['Negativo (0)', 'Positivo (1)'], [contagem_0, contagem_1])
plt.title("Distribuição das Classes")
plt.show()

# ------- SEED + SPLIT

SEED = 42


def set_global_seed(seed: int = SEED):
    """
    Fixa todas as fontes de aleatoriedade do pipeline.
    Chamar antes de qualquer import de TF ou criação de modelo.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)        # hash do Python
    random.seed(seed)                                # módulo random padrão
    import numpy as np
    np.random.seed(seed)                             # NumPy
    import tensorflow as tf
    tf.random.set_seed(seed)                         # TensorFlow / Keras
    # Elimina variações não-determinísticas do cuDNN (custo de ~5-15% de performance)
    tf.config.experimental.enable_op_determinism()


set_global_seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

# ======= PREPARAÇÃO DOS ARRAYS

N = len(dataset)
X = np.zeros((N, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
y = np.array(dataset["label"], dtype=np.int64)

for i in range(N):
    img = dataset[i]["image"].convert("RGB").resize(IMG_SIZE)
    X[i] = np.array(img, dtype=np.float32) / 255.0

# ======= SPLIT DETERMINÍSTICO

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEED,   
    stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=SEED,   
    stratify=y_temp
)


def print_stats(name, y_data):
    count = np.bincount(y_data)
    pct = (count[1] / len(y_data)) * 100
    print(f"{name:<10} | Total: {len(y_data):<5} | Neg(0): {count[0]:<5} "
          f"| Pos(1): {count[1]:<4} | % Pos: {pct:.2f}%")


print("-" * 65)
print(f"{'Conjunto':<10} | {'Total':<6} | {'Classe 0':<8} | {'Classe 1':<7} | Prop. Positiva")
print("-" * 65)
print_stats("Treino",    y_train)
print_stats("Validação", y_val)
print_stats("Teste",     y_test)
print("-" * 65)
print(f"Shape X_train: {X_train.shape}")

# ================ CLASS WEIGHT 
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = dict(zip(classes, weights))
print(f"\nClass weights: {class_weight}")

# ============== Métricas
def get_metrics_with_ci(y_true, y_probs, n_bootstrap=1000, ci=0.95, seed=SEED):
    """
    Calcula AUROC, AUPRC e PPV@90Recall com intervalos de confiança via bootstrap.
    seed: usa SEED global do projeto para reprodutibilidade total.
    """

    def calculate_ppv_at_recall(y_t, y_p, target_recall=0.90):
        precisions, recalls, _ = precision_recall_curve(y_t, y_p)
        idx = np.where(recalls >= target_recall)[0]
        if len(idx) == 0:
            return 0.0
        return float(precisions[idx[-1]])

    boot_auroc, boot_auprc, boot_ppv90 = [], [], []
    print(
        f"Calculando métricas com {n_bootstrap} amostras de bootstrap (seed={seed})...")

    for i in range(n_bootstrap):
        # seed + i → sequência determinística e única por iteração
        y_t_boot, y_p_boot = resample(y_true, y_probs, random_state=seed + i)

        if len(np.unique(y_t_boot)) < 2:
            continue

        boot_auroc.append(roc_auc_score(y_t_boot, y_p_boot))
        boot_auprc.append(average_precision_score(y_t_boot, y_p_boot))
        boot_ppv90.append(calculate_ppv_at_recall(y_t_boot, y_p_boot))

    lower_p = ((1 - ci) / 2) * 100
    upper_p = (ci + (1 - ci) / 2) * 100

    results = {
        "Metric":        ["AUROC", "AUPRC", "PPV@90Recall"],
        "Value":         [
            roc_auc_score(y_true, y_probs),
            average_precision_score(y_true, y_probs),
            calculate_ppv_at_recall(y_true, y_probs),
        ],
        "CI Lower": [
            np.percentile(boot_auroc, lower_p),
            np.percentile(boot_auprc, lower_p),
            np.percentile(boot_ppv90, lower_p),
        ],
        "CI Upper": [
            np.percentile(boot_auroc, upper_p),
            np.percentile(boot_auprc, upper_p),
            np.percentile(boot_ppv90, upper_p),
        ],
    }
    return pd.DataFrame(results)


def full_evaluation_report(y_true, y_probs, threshold=0.5, seed=SEED):
    """
    Relatório completo: classificação, confusão e CIs com bootstrap reprodutível.
    y_probs: saída de model.predict() — aceita shape (N,) ou (N, 1).
    """
    # Garante shape (N,) independente do que o Keras retornar
    y_probs = np.array(y_probs).ravel()
    y_pred = (y_probs >= threshold).astype(int)

    print("-" * 40)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("-" * 40)
    print(classification_report(y_true, y_pred, zero_division=0))

    print("MATRIZ DE CONFUSÃO")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    print(f"\nAcurácia : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_true, y_pred, zero_division=0):.4f}")

    df_ci = get_metrics_with_ci(y_true, y_probs, seed=seed)
    print(f"\nMÉTRICAS COM IC {int(df_ci.shape[0] > 0 and 95)}%")
    print(df_ci.to_string(index=False, float_format="%.4f"))

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


# ============================================================
# DATASETS
# ============================================================
train_ds = build_dataset(X_train, y_train, shuffle=True,  augment=True)
val_ds = build_dataset(X_val,   y_val,   shuffle=False, augment=False)
test_ds = build_dataset(X_test,  y_test,  shuffle=False, augment=False)

# ================================================================
# Balanceamento de Batch
# ================================================================


class BalancedBatchGenerator(Sequence):
    """
    Gera batches com 50% positivo / 50% negativo.
    Reprodutível via seed — cada época embaralha de forma
    determinística mas diferente das anteriores.
    """

    def __init__(self, X, y, batch_size=32, seed=SEED):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.seed = seed
        self.n_per_class = batch_size // 2

        self.indices_pos = np.where(y == 1)[0]
        self.indices_neg = np.where(y == 0)[0]

        # Número de steps: quantas vezes a classe minoritária
        # é coberta com batches balanceados
        self._n_steps = int(np.ceil(len(self.indices_pos) / self.n_per_class))

        # RNG próprio — isolado do estado global do NumPy
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self._n_steps

    def __getitem__(self, index):
        # seed por step e por época para sequência determinística
        step_seed = self.seed + self._epoch * 10000 + index
        rng = np.random.default_rng(step_seed)

        idx_pos = rng.choice(self.indices_pos, self.n_per_class, replace=True)
        idx_neg = rng.choice(self.indices_neg, self.n_per_class, replace=True)

        batch_indices = np.concatenate([idx_pos, idx_neg])
        rng.shuffle(batch_indices)

        return self.X[batch_indices], self.y[batch_indices]

    def on_epoch_end(self):
        """Chamado automaticamente pelo Keras ao fim de cada época."""
        self._epoch += 1

    def on_epoch_begin(self):
        self._epoch = getattr(self, "_epoch", 0)

# ============================================================
# INTEGRAÇÃO COM O AUGMENTATION SELETIVO
# ============================================================
# O generator retorna arrays NumPy por batch — para aplicar
# o augmentation seletivo, convertemos para tf.data via
# from_generator depois de instanciar.


train_generator = BalancedBatchGenerator(
    X_train, y_train, batch_size=BATCH_SIZE, seed=SEED
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

