import numpy as np
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical

DNA_BASES = ["A", "C", "G", "T"]
OHE = OneHotEncoder(handle_unknown="ignore")
OHE.fit(np.array(DNA_BASES).reshape(-1, 1))

SUPERF_DICT = {
    'LTR': 0,
    'COPIA': 1,
    'GYPSY': 2,
    'ERV': 3,
    'BELPAO': 4,
    'LINE': 5,
    'I': 6,
    'L1': 7,
    'RTE': 8,
    'DIRS': 9,
    'PLE': 10,
    'SINE': 11,
    'TRNA': 12,
    'HELITRON': 13,
    'CRYPTON': 14,
    'HAT': 15,
    'MERLIN': 16,
    'P': 17,
    'TIR': 18,
    'TC1MARINER': 19,
    'MULE': 20,
    'PIFHARBINGER': 21,
    'CACTA': 22,
    'PIGGYBAC': 23,
    'CR1': 24,
    'R1': 25,
    'LARD': 26,
    'ALU': 27,
    'KOLOBOK': 28,
    'ACADEM-1': 29
}
NUM_CLASSES = len(SUPERF_DICT)

MAX_EPOCHS = 10
ES_PATIENCE = 50
INITIAL_LR = 0.0002

def seq2oh(seq, len_thre):
    """
    Convierte una secuencia de DNA a one-hot
    """
    list_seq = list(seq)
    if len(list_seq) >= len_thre:
        seq_1 = list_seq[0:len_thre // 2]
        seq_2 = list_seq[-len_thre // 2:]
        final_seq_list = seq_1 + seq_2
    else:
        seq_1 = list_seq[0:len(list_seq) // 2]
        seq_2 = list_seq[-len(list_seq) // 2:]
        padding = ["N"] * (len_thre - len(list_seq))
        final_seq_list = seq_1 + padding + seq_2

    # One-hot
    seq_1_oh = OHE.transform(np.array(seq_1).reshape(-1, 1)).toarray().astype(np.float32)
    seq_2_oh = OHE.transform(np.array(seq_2).reshape(-1, 1)).toarray().astype(np.float32)

    # padding uniforme [0.25, 0.25, 0.25, 0.25]
    pad_len = len_thre - len(list_seq)
    if pad_len > 0:
        padding_oh = np.full((pad_len, 4), 0.25, dtype=np.float32)
        # Concatenacion Part1 + Padding + Part2
        seq_encode = np.vstack([seq_1_oh, padding_oh, seq_2_oh])
    else:
        # No padding
        seq_encode = np.vstack([seq_1_oh, seq_2_oh])

    return seq_encode


def get_label_data(record_id):
    """
    Extrae y codifica la etiqueta de la clase de un ID de FASTA
    """
    try:
        classification = record_id.split("#")[1].split(" ")[0]
        superf = classification.split("/")[2]
        label_idx = SUPERF_DICT[superf]
        return label_idx
    except (IndexError, KeyError):
        return -1


def load_and_preprocess_fasta(file_path, max_len):
    """
    Carga secuencias y etiquetas de un archivo FASTA, las preprocesa y devuelve datos 
    y etiquetas codificados en one-hot
    """
    encoded_data = []
    labels = []
    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq)
        label_idx = get_label_data(record.id)
        if label_idx != -1:
            encoded_data.append(seq2oh(seq, max_len))
            labels.append(label_idx)
    X_train = np.array(encoded_data)
    Y_train_idx = np.array(labels)
    Y_train_oh = to_categorical(Y_train_idx, num_classes=NUM_CLASSES)
    print(f"Secuencias válidas y con etiquetas leídas: {len(X_train)}")
    return X_train, Y_train_oh, NUM_CLASSES


def load_and_preprocess_fasta_grouped(file_path, max_len):
    """
    Carga secuencias y etiquetas de un archivo FASTA, las preprocesa y devuelve datos 
    y etiquetas codificados en one-hot para transposones y retrotransposones.
    """
    elementos_transponibles = {
        "retrotransposones": [
            "GYPSY", "CR1", "ERV", "COPIA", "L1", "RTE", "BELPAO", "I", "LINE",
            "DIRS", "LTR", "PLE", "SINE", "ALU", "R1"
        ],
        "transposones": [
            "TC1MARINER", "HAT", "TIR", "HELITRON", "PIFHARBINGER", "MULE",
            "CACTA", "LARD", "PIGGYBAC", "KOLOBOK", "ACADEM-1", "P", "MERLIN",
            "CRYPTON", "TRNA"
        ]
    }
    encoded_data_transposon = []
    labels_transposon = []
    encoded_data_retrotransposon = []
    labels_retrotransposon = []

    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq)
        label_idx = get_label_data(record.id)
        if label_idx != -1:
            try:
                classification = record.id.split("#")[1].split(" ")[0]
                superf = classification.split("/")[2]
            except (IndexError, KeyError):
                continue

            seq_encoded = seq2oh(seq, max_len)
            if superf in elementos_transponibles["retrotransposones"]:
                encoded_data_retrotransposon.append(seq_encoded)
                labels_retrotransposon.append(label_idx)
            elif superf in elementos_transponibles["transposones"]:
                encoded_data_transposon.append(seq_encoded)
                labels_transposon.append(label_idx)

    X_transposon = np.array(encoded_data_transposon)
    Y_transposon_idx = np.array(labels_transposon)
    Y_transposon_oh = to_categorical(Y_transposon_idx, num_classes=NUM_CLASSES)

    X_retrotransposon = np.array(encoded_data_retrotransposon)
    Y_retrotransposon_idx = np.array(labels_retrotransposon)
    Y_retrotransposon_oh = to_categorical(Y_retrotransposon_idx, num_classes=NUM_CLASSES)

    print(f"Transposones: {len(X_transposon)} secuencias")
    print(f"Retrotransposones: {len(X_retrotransposon)} secuencias")
    return (X_transposon, Y_transposon_oh), (X_retrotransposon, Y_retrotransposon_oh), NUM_CLASSES


BASE_MAP = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


def nucle2num(nucleotide):
    """
    Mapeo de nucleótidos a números
    """
    nucleotide = nucleotide.upper()
    if nucleotide == "A": return 0
    elif nucleotide == "G":
        return 2
    elif nucleotide == "C":
        return 1
    elif nucleotide == "T":
        return 3
    else:
        return -1


def sequence_to_kmer_profile(sequences, k=4):
    """
    Calcula k-mers para una lista de secuencias de ADN
    """
    import numpy as np

    profile_size = 4**k
    counts = np.zeros(profile_size, dtype=np.float32)
    power_k_minus_1 = 4**(k - 1)

    for seq in sequences:
        seq = seq.upper()
        if len(seq) < k:
            continue
        current_hash = 0
        window_valid_len = 0
        for i in range(len(seq)):
            num_in = nucle2num(seq[i])
            if num_in == -1:
                current_hash = 0
                window_valid_len = 0
                continue
            if window_valid_len < k:
                current_hash = current_hash * 4 + num_in
                window_valid_len += 1
                if window_valid_len == k:
                    counts[current_hash] += 1
                continue
            num_out = nucle2num(seq[i - k])
            if num_out == -1:
                current_hash = 0
                window_valid_len = 0
                continue
            current_hash = (current_hash -
                            num_out * power_k_minus_1) * 4 + num_in
            counts[current_hash] += 1

    total_kmers = np.sum(counts)
    if total_kmers == 0:
        return np.zeros(profile_size, dtype=np.float32)
    return counts / total_kmers


def analyze_fasta_distribution(fasta_path):
    """
    COnteo de muestras por clase
    """
    from Bio import SeqIO
    from collections import Counter

    class_counts = Counter()
    for record in SeqIO.parse(fasta_path, "fasta"):
        label_idx = get_label_data(record.id)
        if label_idx != -1:
            class_counts[label_idx] += 1
    return class_counts
