import numpy as np
from Bio import SeqIO

from collections import Counter

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


def get_label_data(record_id):
    """
    Extrae la etiqueta de clase del ID de registro FASTA
    """
    try:
        classification = record_id.split("#")[1].split(" ")[0]
        superf = classification.split("/")[2]
        label_idx = SUPERF_DICT[superf]
        return label_idx
    except (IndexError, KeyError):
        return -1


def analyze_fasta_distribution(fasta_path):
    """
    Cuenta el número de muestras por clase en un archivo FASTA
    """
    class_counts = Counter()

    for record in SeqIO.parse(fasta_path, "fasta"):
        label_idx = get_label_data(record.id)
        if label_idx != -1:
            class_counts[label_idx] += 1

    return class_counts


def data_aug(fasta_file):
    """
    Main
    """
    parser = argparse.ArgumentParser(description="Predict on SENMAP Test Set and Generate Detailed Metrics")
    parser.add_argument("-f", "--fasta_file", required=True, help="Input FASTA file (full dataset)")
    
    args = parser.parse_args()


    records = [r for r in SeqIO.parse(args.fasta_file, "fasta")]
    valid_records = [r for r in records if get_label_data(r.id) != -1]
    class_counts = analyze_fasta_distribution(args.fasta_file)
    augmented_records = []

    for class_idx, count in class_counts.items():
        class_records = [
            r for r in valid_records if get_label_data(r.id) == class_idx
        ]
        n_aug = max(1, int(count * 0.1))  # 10%

        idx_sub = np.random.choice(len(class_records), n_aug, replace=False)
        idx_ins = np.random.choice(len(class_records), n_aug, replace=False)
        idx_del = np.random.choice(len(class_records), n_aug, replace=False)

        # Sustitucion
        for i in idx_sub:
            record = class_records[i]
            seq = str(record.seq)
            seq_len = len(seq)
            num_substitutions = max(1, seq_len // 20)
            seq_list = list(seq)
            for _ in range(num_substitutions):
                pos = np.random.randint(0, seq_len)
                original_base = seq_list[pos]
                bases = ['A', 'C', 'G', 'T']
                if original_base in bases:
                    bases.remove(original_base)
                seq_list[pos] = np.random.choice(bases)
            new_seq_sub = ''.join(seq_list)
            new_record_sub = record[:]
            new_record_sub.seq = new_seq_sub
            augmented_records.append(new_record_sub)

        # Insercion
        for i in idx_ins:
            record = class_records[i]
            seq = str(record.seq)
            seq_len = len(seq)
            num_insertions = max(1, seq_len // 50)
            seq_list = list(seq)
            for _ in range(num_insertions):
                pos = np.random.randint(0, len(seq_list) + 1)
                base_to_insert = np.random.choice(['A', 'C', 'G', 'T'])
                seq_list.insert(pos, base_to_insert)
            new_seq_ins = ''.join(seq_list)
            new_record_ins = record[:]
            new_record_ins.seq = new_seq_ins
            augmented_records.append(new_record_ins)

        # Delecion
        for i in idx_del:
            record = class_records[i]
            seq = str(record.seq)
            seq_len = len(seq)
            num_deletions = max(1, seq_len // 50)
            seq_list = list(seq)
            for _ in range(num_deletions):
                if len(seq_list) == 0:
                    break
                pos = np.random.randint(0, len(seq_list))
                del seq_list[pos]
            new_seq_del = ''.join(seq_list)
            new_record_del = record[:]
            new_record_del.seq = new_seq_del
            augmented_records.append(new_record_del)

    output_path = args.fasta_file.replace(".fasta", "_augmented.fasta")

    all_records = records + augmented_records
    with open(output_path, "w") as out_fasta:
        SeqIO.write(all_records, out_fasta, "fasta")


if __name__ == "__main__":
    data_aug()
