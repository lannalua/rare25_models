#importação do dataset rare25 via kaggle
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
