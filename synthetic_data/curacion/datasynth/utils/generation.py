from Bio import SeqIO, Entrez
import random
import os
import multiprocessing
from pathlib import Path
import time
import sys
from types import SimpleNamespace download_microsatellites_from_ncbi, download_genes_from_ncbi

from download import 

def generate_realistic_dna(length, gc_content=None):
    """
    Genera ADN sintético
    """
    if length <= 0:
        return ""

    if gc_content is None:
        gc_content = random.uniform(0.35, 0.55)

    gc_count = int(length * gc_content)
    at_count = length - gc_count

    g_count = gc_count // 2
    c_count = gc_count - g_count
    a_count = at_count // 2
    t_count = at_count - a_count

    nucleotides = (['g'] * g_count) + (['c'] * c_count) + (['a'] * a_count) + (
        ['t'] * t_count)
    random.shuffle(nucleotides)

    return ''.join(nucleotides[:length])


def generate_realistic_microsatellite(target_length,
                                      imperfect_prob=0.15,
                                      real_microsat_sequences=None):
    """
    Genera microsatélites
    """
    if target_length <= 0:
        return ""

    if real_microsat_sequences and len(real_microsat_sequences) > 0:
        selected_seq = random.choice(real_microsat_sequences)
        seq_str = str(selected_seq.seq).lower().replace('n', '')

        if not seq_str:
            return generate_realistic_dna(target_length)
    else:
        motif_options = ['a', 't', 'g', 'c', 'at', 'gt', 'ca', 'tg']
        motif = random.choice(motif_options)
        seq_str = motif * (1000 // len(motif))

    result = ""
    while len(result) < target_length:
        result += seq_str
    result = result[:target_length]

    if imperfect_prob > 0:
        result_list = list(result)
        for i in range(len(result_list)):
            if random.random() < imperfect_prob:
                current = result_list[i]
                alternatives = [
                    n for n in ['a', 't', 'c', 'g'] if n != current
                ]
                if alternatives:
                    result_list[i] = random.choice(alternatives)
        result = ''.join(result_list)

    return result.lower()


def get_te_fragment(te_sequence,
                    fragmentation_prob=0.3,
                    min_frac=0.5,
                    max_frac=0.95):
    """
    Retorna un TE completo o fragmentado
    """
    te_sequence = str(te_sequence).lower()

    if random.random() < fragmentation_prob and len(te_sequence) > 1000:
        fragment_length = int(
            len(te_sequence) * random.uniform(min_frac, max_frac))
        fragment_length = min(fragment_length, len(te_sequence))
        max_start = len(te_sequence) - fragment_length
        if max_start < 0:
            max_start = 0
        start = random.randint(0, max_start)
        return te_sequence[start:start + fragment_length], True
    return te_sequence, False


def extract_species_name(seq_record):
    """
    Extrae el nombre de la especie de la descripción del registro
    """
    description = seq_record.description
    parts = description.split()
    if len(parts) > 1:
        return parts[1].replace("#", "_")
    return "unknown_species"


# ----------------- CASOS -----------------

# Caso 1 (ADN + TE1 + ADN + TE2 + ADN)
def generate_case1(sequences, total_length, config, real_microsat_sequences):
    """
    Caso 1: ADN + TE1 + ADN + TE2 + ADN
    """
    seq1_choice = random.choice(sequences)
    seq2_choice = random.choice(sequences)

    seq1_str, seq1_frag = get_te_fragment(str(seq1_choice.seq),
                                          fragmentation_prob=config.get(
                                              'fragmentation_prob', 0.3))
    seq2_str, seq2_frag = get_te_fragment(str(seq2_choice.seq),
                                          fragmentation_prob=config.get(
                                              'fragmentation_prob', 0.3))

    seq1_len = len(seq1_str)
    seq2_len = len(seq2_str)

    # flanqueo
    min_flanking_te = 100

    len_total_te = seq1_len + seq2_len
    len_remaining = total_length - len_total_te

    if len_remaining < min_flanking_te * 3:
        len_to_keep = int(total_length * 0.3)
        if len_to_keep < 100: len_to_keep = 100
        seq1_str = seq1_str[:len_to_keep]
        seq2_str = seq2_str[:len_to_keep]
        seq1_len = len(seq1_str)
        seq2_len = len(seq2_str)
        len_total_te = seq1_len + seq2_len
        len_remaining = total_length - len_total_te

    if len_remaining < 3:
        len_adn1 = len_adn2 = len_adn3 = len_remaining // 3
    else:
        points = sorted(random.sample(range(1, len_remaining), 2))
        len_adn1 = points[0]
        len_adn2 = points[1] - points[0]
        len_adn3 = len_remaining - points[1]

    final_seq = generate_realistic_dna(len_adn1)
    final_seq += seq1_str
    final_seq += generate_realistic_dna(len_adn2)
    final_seq += seq2_str
    final_seq += generate_realistic_dna(len_adn3)

    final_seq = final_seq[:total_length].lower()

    start_pos_1 = len_adn1
    end_pos_2 = len_adn1 + seq1_len + len_adn2 + seq2_len

    seq_id = seq1_choice.id.replace('#', '_') + "_" + seq2_choice.id.replace(
        '#', '_')
    seq_species = extract_species_name(seq1_choice)
    frag_label = "frag" if (seq1_frag or seq2_frag) else "full"

    return f">Caso1_{frag_label}_{seq_id}_{start_pos_1}_{end_pos_2}_{seq_species}\n{final_seq}"


# Caso 2 (TE1 + TE2 + TE1)
def generate_case2(sequences, total_length, config, real_microsat_sequences):
    """
    Caso 2: TE1 + TE2 + TE1 (Repetición)
    """
    min_length_filter = 5000
    filtered_sequences = [
        seq for seq in sequences if len(seq) > min_length_filter
    ]
    current_sequences = filtered_sequences if filtered_sequences else sequences

    len_obj = total_length // 3

    seq1_choice = random.choice(current_sequences)
    seq1_species = extract_species_name(seq1_choice)

    seq1_seq, seq1_frag = get_te_fragment(
        str(seq1_choice.seq), config.get('fragmentation_prob', 0.3))

    sequence_1 = seq1_seq[:len_obj] if len(
        seq1_seq) >= len_obj else seq1_seq + generate_realistic_dna(
            len_obj - len(seq1_seq))

    seq2_choice = random.choice(current_sequences)
    seq2_seq, seq2_frag = get_te_fragment(
        str(seq2_choice.seq), config.get('fragmentation_prob', 0.3))

    sequence_2 = seq2_seq[:len_obj] if len(
        seq2_seq) >= len_obj else seq2_seq + generate_realistic_dna(
            len_obj - len(seq2_seq))

    final_seq = sequence_1 + sequence_2 + sequence_1

    if len(final_seq) < total_length:
        final_seq += generate_realistic_dna(total_length - len(final_seq))

    final_seq = final_seq[:total_length].lower()

    start_pos = 0
    end_pos = len(sequence_1)

    frag_label = "frag" if (seq1_frag or seq2_frag) else "full"

    return f">Caso2_{frag_label}_{seq1_choice.id.replace('#', '_')}_0_{end_pos}_{seq1_species.replace('#', '_')}\n{final_seq}"


# Caso 3 (Microsatélite + TE + Microsatélite)
def generate_case3(sequences, total_length, config, real_microsat_sequences):
    """
    Caso 3: Microsatélite + TE + Microsatélite
    """
    random_sequence = random.choice(sequences)
    seq_species = extract_species_name(random_sequence)

    sequence_str, seq_frag = get_te_fragment(str(random_sequence.seq),
                                             fragmentation_prob=config.get(
                                                 'fragmentation_prob', 0.3))
    sequence_length = len(sequence_str)

    remaining_for_microsat = total_length - sequence_length
    if remaining_for_microsat < 0:
        sequence_str = sequence_str[:total_length // 2]
        sequence_length = len(sequence_str)
        remaining_for_microsat = total_length - sequence_length

    before_length = remaining_for_microsat // 2
    after_length = remaining_for_microsat - before_length

    microsat_before = generate_realistic_microsatellite(
        before_length,
        imperfect_prob=config.get('microsat_imperfect_prob', 0.15),
        real_microsat_sequences=real_microsat_sequences)
    microsat_after = generate_realistic_microsatellite(
        after_length,
        imperfect_prob=config.get('microsat_imperfect_prob', 0.15),
        real_microsat_sequences=real_microsat_sequences)

    final_seq = microsat_before + sequence_str + microsat_after
    final_seq = final_seq[:total_length].lower()

    start_pos = len(microsat_before)
    end_pos = start_pos + sequence_length
    if end_pos > total_length: end_pos = total_length

    frag_label = "frag" if seq_frag else "full"

    return f">Caso3_{frag_label}_{random_sequence.id.replace('#', '_')}_{start_pos}_{end_pos}_{seq_species.replace('#', '_')}\n{final_seq}"


# Caso 4 (Falso Positivo TE - Microsatelites complejos)
def generate_case4(sequences, total_length, config, real_microsat_sequences):
    """
    Caso 4: Secuencia con microsatélites complejos/repetitivos 
    que simulan un 'Falso Positivo' de TE.
    """

    # region central altamente repetitiva
    false_te_len = random.randint(1000, 5000)

    false_te_motif = generate_realistic_microsatellite(
        false_te_len,
        imperfect_prob=config.get('microsat_imperfect_prob', 0.05),
        real_microsat_sequences=real_microsat_sequences)

    # flanqueo
    remaining_length = total_length - false_te_len
    if remaining_length < 0: remaining_length = 0

    before_length = remaining_length // 2
    after_length = remaining_length - before_length

    # ADN flanqueante
    flank_before = generate_realistic_dna(before_length,
                                          gc_content=random.uniform(0.55, 0.7))
    flank_after = generate_realistic_dna(after_length,
                                         gc_content=random.uniform(0.55, 0.7))

    # ADN RUIDOSO + FALSO TE + ADN RUIDOSO
    final_seq = flank_before + false_te_motif + flank_after
    final_seq = final_seq[:total_length].lower()

    start_pos = 0
    end_pos = 1

    random_te = random.choice(sequences)
    seq_species = extract_species_name(random_te)

    return f">Caso4_FalseTE_{random_te.id.replace('#', '_')}_{start_pos}_{end_pos}_{seq_species.replace('#', '_')}\n{final_seq}"


# Generador de Caso 5 (Falso Positivo TE - Gen de Copia Múltiple)
def generate_case5(sequences, total_length, config, real_multicopy_genes):
    """
    Caso 5: Secuencia con Gen de Copia Múltiple (e.g., rDNA) que se confunde con TE.
    """
    if not real_multicopy_genes:
        return generate_case1(sequences, total_length, config, None)

    # seleccionar gen repetitivo
    gene_choice = random.choice(real_multicopy_genes)
    gene_str = str(gene_choice.seq).lower()
    gene_len = len(gene_str)

    # fragmentar gen
    frag_prob = config.get('fragmentation_prob', 0.5)
    gene_fragment, is_frag = get_te_fragment(gene_str,
                                             fragmentation_prob=frag_prob,
                                             min_frac=0.3,
                                             max_frac=0.8)
    gene_frag_len = len(gene_fragment)

    # calcular flanqueo
    min_flanking = 500
    len_remaining = total_length - gene_frag_len

    if len_remaining < min_flanking * 2:
        # Truncar si es necesario
        gene_fragment = gene_fragment[:total_length - min_flanking * 2]
        gene_frag_len = len(gene_fragment)
        len_remaining = total_length - gene_frag_len

    before_length = random.randint(min_flanking, len_remaining - min_flanking)
    after_length = len_remaining - before_length

    # flanqueo
    flank_before = generate_realistic_dna(before_length,
                                          gc_content=random.uniform(
                                              0.35, 0.55))
    flank_after = generate_realistic_dna(after_length,
                                         gc_content=random.uniform(0.35, 0.55))

    final_seq = flank_before + gene_fragment + flank_after
    final_seq = final_seq[:total_length].lower()

    # labels
    start_pos = len(flank_before)
    end_pos = start_pos + gene_frag_len

    random_te = random.choice(sequences)
    seq_species = extract_species_name(random_te)
    frag_label = "frag" if is_frag else "full"

    return f">Caso5_MulticopyGene_{frag_label}_{gene_choice.id}_{start_pos}_{end_pos}_{seq_species.replace('#', '_')}\n{final_seq}"


def generate_simulated_data_wrapper(case, sequences, total_length, config,
                                    real_microsat_sequences,
                                    real_multicopy_genes):
    """
    Wrapper para llamar al generador del caso especifico
    """
    try:
        if case == 1:
            return generate_case1(sequences, total_length, config,
                                  real_microsat_sequences)
        elif case == 2:
            return generate_case2(sequences, total_length, config,
                                  real_microsat_sequences)
        elif case == 3:
            return generate_case3(sequences, total_length, config,
                                  real_microsat_sequences)
        elif case == 4:
            return generate_case4(sequences, total_length, config,
                                  real_microsat_sequences)
        elif case == 5:
            return generate_case5(sequences, total_length, config,
                                  real_multicopy_genes)
        else:
            raise ValueError(f"Caso {case} no válido")
    except Exception as e:
        print(f"Error generando Caso {case}: {e}", file=sys.stderr)
        return None


def generation_multiprocessing(sequences, n, output_file, config,
                               real_microsat_sequences, total_length,
                               real_multicopy_genes):
    """
    Genera n secuencias sintéticas por caso y las escribe en un archivo
    """
    results = []

    for case in range(1, 6):
        print("    - Generando Caso", case)
        for _ in range(n):
            result = generate_simulated_data_wrapper(
                case,
                sequences,
                total_length=total_length,
                config=config,
                real_microsat_sequences=real_microsat_sequences,
                real_multicopy_genes=real_multicopy_genes)
            if result:
                results.append(result)

    with open(output_file, "w") as simulated_data:
        for r in results:
            simulated_data.write(r + "\n")


def generate_synthetic_datasets(email, fragmentation_prob,
                                microsat_imperfect_prob, fasta_url,
                                seq_per_case, processes, output_dir,
                                total_length):
    """
    Función principal llamada por el script externo para generar el dataset sintético.
    """
    # email, fragmentation_prob, microsat_imperfect_prob, fasta_url, seq_per_case, processes, output_dir

    global Entrez
    Entrez.email = email

    config = {
        'fragmentation_prob': fragmentation_prob,
        'microsat_imperfect_prob': microsat_imperfect_prob
    }

    # descarga microsatélites
    real_microsat_sequences = None
    microsat_path = os.path.join(output_dir, "ncbi_microsats.fasta")

    real_microsat_sequences = download_microsatellites_from_ncbi(
        output_file=microsat_path)
    if not real_microsat_sequences:
        print("Error: no se pudieron cargar microsatélites")

    # descarga genes de copia multiple
    multicopy_gene_file = os.path.join(output_dir,
                                       "ncbi_multicopy_genes.fasta")

    real_multicopy_genes = download_genes_from_ncbi(
        output_file=multicopy_gene_file)
    if not real_multicopy_genes:
        print("WARNING: no se pudieron cargar genes de copia múltiple")

    # carga secuencias de TE
    try:
        with open(fasta_url, "r") as fasta_file:
            sequences = list(SeqIO.parse(fasta_file, "fasta"))
    except Exception as e:
        raise ValueError(
            f"Error al leer o parsear el archivo FASTA {fasta_url}: {e}")

    if not sequences:
        raise ValueError(
            f"No se encontraron secuencias de TE válidas en {fasta_url}")

    total_seqs_per_case = seq_per_case
    seqs_per_process = total_seqs_per_case // processes

    output_base_name = "synthetic_dataset_mixed"

    for i in range(processes):
        temp_file = os.path.join(output_dir,
                                 f"{output_base_name}_part{i + 1}.fasta")
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print(
        f"Generando {total_seqs_per_case} secuencias por caso (x4 casos) con {processes} procesos..."
    )

    processes_list = []

    for i in range(processes):
        start_idx = i * seqs_per_process
        end_idx = start_idx + seqs_per_process

        if i == processes - 1:
            end_idx = total_seqs_per_case

        num_seqs_for_process = end_idx - start_idx

        if num_seqs_for_process <= 0:
            continue

        output_file_part = os.path.join(
            output_dir, f"{output_base_name}_part{i + 1}.fasta")
        p = multiprocessing.Process(
            target=generation_multiprocessing,
            args=(sequences, num_seqs_for_process, output_file_part, config,
                  real_microsat_sequences, total_length, real_multicopy_genes))
        processes_list.append(p)
        p.start()

    for p in processes_list:
        p.join()

    # fusiona archivos temporales
    merged_output_file = os.path.join(output_dir, f"{output_base_name}.fasta")
    total_sequences_generated = 0

    with open(merged_output_file, "w") as outfile:
        for i in range(processes):
            part_file = os.path.join(output_dir,
                                     f"{output_base_name}_part{i + 1}.fasta")
            if os.path.exists(part_file):
                with open(part_file, "r") as infile:
                    content = infile.read()
                    outfile.write(content)
                    total_sequences_generated += content.count('>')
                os.remove(part_file)
            else:
                print(f"Archivo temporal no encontrado: {part_file}")
