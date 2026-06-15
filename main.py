#importação do dataset rare25 via kaggle
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


# ---------- Split do dataset

# Configurações iniciais
IMG_SIZE = (224, 224)
N = len(dataset)

# 1. Preparação dos arrays (X e y)
X = np.zeros((N, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
y = np.array(dataset["label"], dtype=np.int64)

for i in range(N):
    # Converte para RGB para garantir 3 canais e redimensiona
    img = dataset[i]["image"].convert("RGB").resize(IMG_SIZE)
    # Normalização para o intervalo [0, 1]
    X[i] = np.array(img, dtype=np.float32) / 255.0

# 2. Primeiro Split: Separa Treino (80%) e o restante (20%)
# O stratify=y garante que os 5% de casos de câncer sejam distribuídos proporcionalmente
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Segundo Split: Divide os 20% restantes ao meio (10% Validação e 10% Teste)
# Usamos stratify=y_temp para manter a proporção nos novos subconjuntos
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 4. Verificação dos resultados


def print_stats(name, y_data):
    count = np.bincount(y_data)
    percent_pos = (count[1] / len(y_data)) * 100
    print(
        f"{name:<10} | Total: {len(y_data):<5} | Neg(0): {count[0]:<5} | Pos(1): {count[1]:<4} | % Pos: {percent_pos:.2f}%")


print("-" * 65)
print(f"{'Conjunto':<10} | {'Total':<6} | {'Classe 0':<8} | {'Classe 1':<7} | {'Prop. Positiva'}")
print("-" * 65)
print_stats("Treino", y_train)
print_stats("Validação", y_val)
print_stats("Teste", y_test)
print("-" * 65)

# Verificação de shape para conferir se as dimensões estão corretas para o modelo
print(f"Shape final do X_train: {X_train.shape}")
