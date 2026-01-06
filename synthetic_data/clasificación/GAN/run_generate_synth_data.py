"""
Evalúa límites óptimos de aumento sintético por clase y especie usando JSD
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from Bio import SeqIO
from Bio.Seq import Seq
from math import log2
import os
import sys
import csv
import argparse
from collections import Counter
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from utils.utils import (get_label_data, SUPERF_DICT, NUM_CLASSES, DNA_BASES,
                         sequence_to_kmer_profile, nucle2num,
                         analyze_fasta_distribution)
from utils.models import make_generator_model

from utils.utils_synth_data import (generate_synthetic_sequences,
                                    jensen_shannon_divergence,
                                    get_real_sequences_for_class,
                                    extract_species_id,
                                    get_species_data,
                                    kneedle_knee_point,
                                    plot_pca)

KMER_SIZE = 4
BATCH_SIZE_INCREMENT = 1000
MAX_SYNTHETIC_EVAL = 15000
NOISE_DIM = 100
SEQ_LEN = 600
CHANNELS = 4
OUTPUT_CSV = 'optimal_synthetic_limits.csv'

JSD_SMOOTH_WINDOW = 3
STOP_PATIENCE = 2
JSD_EPS = 1e-3
MIN_GAIN_EPS = 5e-4
MIN_STEPS_BEFORE_STOP = 4


def get_args():
    """
    Parse command-line arguments for synthetic data generation.
    """
    parser = argparse.ArgumentParser(
        description=
        "genera datos sintéticos cGAN usando límites calculados por JSD")
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='archivo FASTA')
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help=
        'ruta al modelo entrenado'
    )
    parser.add_argument(
        '--is_grouping',
        action='store_true',
        help=
        "Indica si se usan modelos separados para retrotransposones y transposones"
    )

    return parser.parse_args()


def find_optimal_augmentation_limit_per_species(generator,
                                                fasta_file,
                                                species_id,
                                                species_sequences,
                                                species_class_counts,
                                                per_step_batch=500):
    """
    Calcula el óptimo por especie con curva JSD y detección de codo
    """
    real_sequences = species_sequences
    if len(real_sequences) == 0:
        return 0, {}

    real_kmer_profile = sequence_to_kmer_profile(real_sequences, k=KMER_SIZE)

    class_counts = species_class_counts.copy()
    inv_freq = {c: 1.0 / max(1, class_counts[c]) for c in class_counts}
    s = sum(inv_freq.values()) or 1.0
    weights = {c: inv_freq[c] / s for c in class_counts}

    jsd_history = []
    synth_count_history = []
    generated_per_class = Counter()
    current_synthetic_records = []

    max_steps = max(1, MAX_SYNTHETIC_EVAL // per_step_batch)

    for _ in range(max_steps):
        batch_alloc = {}
        for cls, w in weights.items():
            batch_alloc[cls] = int(round(w * per_step_batch))
        diff = per_step_batch - sum(batch_alloc.values())
        if diff != 0:
            for cls, _ in sorted(weights.items(), key=lambda kv: -kv[1]):
                if diff == 0:
                    break
                batch_alloc[cls] += 1 if diff > 0 else -1
                diff += -1 if diff > 0 else 1

        for cls, n in batch_alloc.items():
            if n <= 0:
                continue
            new_records = generate_synthetic_sequences(generator, cls, n)
            current_synthetic_records.extend(new_records)
            generated_per_class[cls] += n

        synth_sequences = [str(r.seq) for r in current_synthetic_records]
        combined_sequences = real_sequences + synth_sequences
        combined_kmer_profile = sequence_to_kmer_profile(combined_sequences,
                                                         k=KMER_SIZE)
        jsd = jensen_shannon_divergence(real_kmer_profile,
                                        combined_kmer_profile)
        jsd_history.append(jsd)
        synth_count_history.append(len(current_synthetic_records))

    if not jsd_history:
        return 0, {}

    idx = kneedle_knee_point(synth_count_history, jsd_history)
    optimal_total = synth_count_history[idx]

    total_generated = sum(generated_per_class.values()) or 1
    per_class_limits = {}
    for cls, cnt in generated_per_class.items():
        per_class_limits[cls] = int(
            round(optimal_total * (cnt / total_generated)))

    # --- PCA & JSD final
    final_synth_records = current_synthetic_records[:optimal_total]
    final_synth_seqs = [str(r.seq) for r in final_synth_records]
    
    if final_synth_seqs:
        real_prof = sequence_to_kmer_profile(real_sequences, k=KMER_SIZE)
        synth_prof = sequence_to_kmer_profile(final_synth_seqs, k=KMER_SIZE)
        combined_seqs = real_sequences + final_synth_seqs
        combined_prof = sequence_to_kmer_profile(combined_seqs, k=KMER_SIZE)
        final_jsd = jensen_shannon_divergence(real_prof, combined_prof)
        print(f"------------ FIN: JSD Final (Real vs Real+Synth): {final_jsd:.6f}")

        os.makedirs('pca_plots', exist_ok=True)
        plot_filename = f"pca_plots/pca_{species_id}_optimal.png"
        plot_pca(real_sequences, final_synth_seqs, f"Species {species_id} (Opt: {optimal_total})", plot_filename)

    return optimal_total, per_class_limits


def find_optimal_augmentation_limit(generator, fasta_file, target_class_idx,
                                    max_count):
    """Itera generando y midiendo JSD para hallar un punto de parada robusto."""
    real_sequences = get_real_sequences_for_class(fasta_file, target_class_idx)
    current_total_count = len(real_sequences)

    real_kmer_profile = sequence_to_kmer_profile(real_sequences, k=KMER_SIZE)

    current_synthetic_records = []
    jsd_history = []
    smooth_jsd_history = []
    synth_count_history = []
    best_smooth_jsd = None
    best_index = -1

    inv_superf_dict = {v: k for k, v in SUPERF_DICT.items()}
    class_name = inv_superf_dict[target_class_idx]

    print(
        f"\n--- Evaluando Límite para Clase: {class_name} (Inicial: {current_total_count}) ---"
    )

    target_ratio = float(os.environ.get('TARGET_RATIO', '0.8'))
    per_class_cap = int(os.environ.get('PER_CLASS_CAP', '10000'))
    target_per_class = min(int(target_ratio * max_count),
                           current_total_count + per_class_cap)

    majority_gap = max(0, target_per_class - current_total_count)
    limit_eval = min(MAX_SYNTHETIC_EVAL, 3 * current_total_count, majority_gap)

    if current_total_count == 0:
        print("Clase sin muestras reales. No se puede calcular JSD.")
        return 0
    if limit_eval == 0:
        print("La clase ya alcanza el objetivo fijado. Límite óptimo: 0")
        return 0

    total_generated = 0
    while total_generated < limit_eval:
        remaining_gap = limit_eval - total_generated
        batch_size = min(BATCH_SIZE_INCREMENT, remaining_gap)

        new_records = generate_synthetic_sequences(generator, target_class_idx,
                                                   batch_size)
        current_synthetic_records.extend(new_records)
        total_generated += batch_size

        synth_sequences = [str(r.seq) for r in current_synthetic_records]
        combined_sequences = real_sequences + synth_sequences

        combined_kmer_profile = sequence_to_kmer_profile(combined_sequences,
                                                         k=KMER_SIZE)

        jsd = jensen_shannon_divergence(real_kmer_profile,
                                        combined_kmer_profile)
        jsd_history.append(jsd)
        synth_count_history.append(total_generated)

        w = max(1, JSD_SMOOTH_WINDOW)
        if len(jsd_history) < w:
            smooth_jsd = float(np.mean(jsd_history))
        else:
            smooth_jsd = float(np.mean(jsd_history[-w:]))
        smooth_jsd_history.append(smooth_jsd)

        if best_smooth_jsd is None or smooth_jsd < best_smooth_jsd - JSD_EPS:
            best_smooth_jsd = smooth_jsd
            best_index = len(smooth_jsd_history) - 1

        stop_due_to_rise = False
        stop_due_to_low_gain = False

        if len(smooth_jsd_history) >= max(MIN_STEPS_BEFORE_STOP,
                                          STOP_PATIENCE + 1):
            rising = True
            for i in range(1, STOP_PATIENCE + 1):
                if not (smooth_jsd_history[-i] - smooth_jsd_history[-i - 1]
                        > JSD_EPS):
                    rising = False
                    break
            stop_due_to_rise = rising

            improvement = smooth_jsd_history[-(STOP_PATIENCE +
                                               1)] - smooth_jsd_history[-1]
            stop_due_to_low_gain = improvement < MIN_GAIN_EPS

        if stop_due_to_rise or stop_due_to_low_gain:
            optimal_idx = best_index if best_index >= 0 else len(
                smooth_jsd_history) - 1
            optimal_limit = synth_count_history[optimal_idx]
            reason = "aumento sostenido" if stop_due_to_rise else "ganancia marginal insuficiente"

            # Evitar detener en el primer batch si hay margen suficiente
            if optimal_limit <= BATCH_SIZE_INCREMENT and limit_eval >= 2 * BATCH_SIZE_INCREMENT:
                print("Demasiado pronto para parar; esperando más evidencia.")
                continue

            remaining_gap = limit_eval - total_generated
            if remaining_gap > 0:
                confirm_batch = min(BATCH_SIZE_INCREMENT, remaining_gap)
                new_records_confirm = generate_synthetic_sequences(
                    generator, target_class_idx, confirm_batch)
                confirm_records = current_synthetic_records + new_records_confirm

                synth_sequences_c = [str(r.seq) for r in confirm_records]
                combined_sequences_c = real_sequences + synth_sequences_c
                combined_kmer_profile_c = sequence_to_kmer_profile(
                    combined_sequences_c, k=KMER_SIZE)
                jsd_c = jensen_shannon_divergence(real_kmer_profile,
                                                  combined_kmer_profile_c)

                jsd_hist_c = jsd_history + [jsd_c]
                if len(jsd_hist_c) < w:
                    smooth_c = float(np.mean(jsd_hist_c))
                else:
                    smooth_c = float(np.mean(jsd_hist_c[-w:]))

                if best_smooth_jsd is None or smooth_c < best_smooth_jsd - JSD_EPS:
                    optimal_limit = total_generated + confirm_batch
                    print(
                        f"Lote extra aceptado: JSD_smooth mejora ({smooth_c:.4f} < {best_smooth_jsd if best_smooth_jsd is not None else float('inf'):.4f})."
                    )
                else:
                    print(
                        f"Lote extra descartado: no mejora JSD_smooth ({smooth_c:.4f} >= {best_smooth_jsd:.4f})."
                    )

            print(f"FIN: {reason}. Límite óptimo: {optimal_limit}")
            
            # --- PCA & JSD Final Report ---
            final_synth_records = current_synthetic_records[:optimal_limit]
            final_synth_seqs = [str(r.seq) for r in final_synth_records]
            
            if final_synth_seqs:
                real_prof = sequence_to_kmer_profile(real_sequences, k=KMER_SIZE)
                # JSD vs Combined is what we tracked, but let's report it clearly
                combined_seqs = real_sequences + final_synth_seqs
                combined_prof = sequence_to_kmer_profile(combined_seqs, k=KMER_SIZE)
                final_jsd = jensen_shannon_divergence(real_prof, combined_prof)
                print(f"------ RESULT: JSD Final (Real vs Real+Synth): {final_jsd:.6f}")
                
                # Plot PCA
                os.makedirs('pca_plots', exist_ok=True)
                plot_filename = f"pca_plots/pca_class_{target_class_idx}_optimal.png"
                plot_pca(real_sequences, final_synth_seqs, f"Class {target_class_idx} (Opt: {optimal_limit})", plot_filename)

            return optimal_limit

        print(
            f"   Generados: {total_generated} | Total: {current_total_count + total_generated} | JSD={jsd:.4f} | JSD_smooth={smooth_jsd:.4f}"
        )

    # Limit eval (FIN)
    optimal_limit = synth_count_history[-1] if synth_count_history else 0
    
    final_synth_records = current_synthetic_records[:optimal_limit]
    final_synth_seqs = [str(r.seq) for r in final_synth_records]
    
    if final_synth_seqs:
        real_prof = sequence_to_kmer_profile(real_sequences, k=KMER_SIZE)
        combined_seqs = real_sequences + final_synth_seqs
        combined_prof = sequence_to_kmer_profile(combined_seqs, k=KMER_SIZE)
        final_jsd = jensen_shannon_divergence(real_prof, combined_prof)
        print(f"--------------- JSD Final (Real vs Real+Synth): {final_jsd:.6f}")
        
        os.makedirs('pca_plots', exist_ok=True)
        plot_filename = f"pca_plots/pca_class_{target_class_idx}_limit.png"
        plot_pca(real_sequences, final_synth_seqs, f"Class {target_class_idx} (Limit: {optimal_limit})", plot_filename)

    return optimal_limit


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

if __name__ == "__main__":
    args = get_args()
    fasta_file = args.data_file
    model_name = args.model_name
    if not args.is_grouping:
        model_path = os.path.join('trained_models', model_name)

        # cargar el Generador
        try:
            generator = make_generator_model(seq_len=SEQ_LEN,
                                             channels=CHANNELS,
                                             noise_dim=NOISE_DIM,
                                             num_classes=NUM_CLASSES)
            generator.load_weights(model_path)
            print("Modelo Generador cargado.")
        except Exception as e:
            print(f"[ERROR] Error al cargar el modelo. {e}")
            sys.exit(1)

        # analizar clases
        class_counts = analyze_fasta_distribution(fasta_file)
        max_count = max(class_counts.values())
        inv_superf_dict = {v: k for k, v in SUPERF_DICT.items()}

        optimal_limits_list = []

        use_species_knee = os.environ.get('USE_SPECIES_KNEE', '0') == '1'
        species_regex = os.environ.get('SPECIES_REGEX', '')

        if use_species_knee:
            species_to_sequences, species_to_class_counts = get_species_data(
                fasta_file, species_regex)
            for species_id, seqs in species_to_sequences.items():
                print(f"\n=== Especie: {species_id} | reales: {len(seqs)} ===")
                optimal_total, per_class_limits = find_optimal_augmentation_limit_per_species(
                    generator,
                    fasta_file,
                    species_id,
                    seqs,
                    species_to_class_counts.get(species_id, Counter()),
                    per_step_batch=int(os.environ.get('SPECIES_BATCH', '500')))
                print(
                    f"   [KNEE] Óptimo por especie {species_id}: {optimal_total} sintéticas"
                )
                for class_idx, limit in per_class_limits.items():
                    current_count = class_counts.get(class_idx, 0)
                    optimal_limits_list.append({
                        'class_name':
                        inv_superf_dict[class_idx],
                        'class_idx':
                        class_idx,
                        'original_count':
                        current_count,
                        'synthetic_limit':
                        limit
                    })
        else:
            # evaluar el límite óptimo para cada clase minoritaria
            total_samples = sum(class_counts.values())
            for class_idx, current_count in class_counts.items():
                class_percentage = (current_count / total_samples
                                    ) * 100 if total_samples > 0 else 0

                if current_count < max_count and class_percentage < 5.0:
                    optimal_synth_count = find_optimal_augmentation_limit(
                        generator, fasta_file, class_idx, max_count)

                    optimal_limits_list.append({
                        'class_name':
                        inv_superf_dict[class_idx],
                        'class_idx':
                        class_idx,
                        'original_count':
                        current_count,
                        'synthetic_limit':
                        optimal_synth_count
                    })
                    print(
                        f"\n--- Límite ÓPTIMO para Clase {inv_superf_dict[class_idx]}: {optimal_synth_count} muestras sintéticas ---"
                    )
                elif class_percentage >= 5.0:
                    print(
                        f"\n--- Clase {inv_superf_dict[class_idx]} tiene {class_percentage:.2f}% del total. No se evalúa augmentation limit. ---"
                    )
                else:
                    print(
                        f"\n--- Clase {inv_superf_dict[class_idx]} ya es mayoritaria ({current_count}). Saltando evaluación. ---"
                    )

        if optimal_limits_list:
            with open(OUTPUT_CSV, 'w', newline='') as csvfile:
                fieldnames = [
                    'class_name', 'class_idx', 'original_count',
                    'synthetic_limit'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(optimal_limits_list)

            print(f"\nLimites optimos guardados en: {OUTPUT_CSV}")
            
    else:

        model_path_retro = os.path.join('trained_models',
                                        f'retrotransposons_{model_name}')
        model_path_trans = os.path.join('trained_models',
                                        f'transposons_{model_name}')
        #Cargar el Generador
        try:
            generator_retro = make_generator_model(seq_len=SEQ_LEN,
                                                   channels=CHANNELS,
                                                   noise_dim=NOISE_DIM,
                                                   num_classes=NUM_CLASSES)
            generator_retro.load_weights(model_path_retro)

            generator_trans = make_generator_model(seq_len=SEQ_LEN,
                                                   channels=CHANNELS,
                                                   noise_dim=NOISE_DIM,
                                                   num_classes=NUM_CLASSES)
            generator_trans.load_weights(model_path_trans)
            print("Modelo Generador cargado.")
        except Exception as e:
            print(f"[ERROR] Error al cargar el modelo. {e}")
            sys.exit(1)

        # Analizar clases
        class_counts = analyze_fasta_distribution(fasta_file)
        print(class_counts)
        max_count = max(class_counts.values())
        inv_superf_dict = {v: k for k, v in SUPERF_DICT.items()}

        optimal_limits_list = []

        use_species_knee = os.environ.get('USE_SPECIES_KNEE', '0') == '1'
        species_regex = os.environ.get('SPECIES_REGEX', '')

        if use_species_knee:
            species_to_sequences, species_to_class_counts = get_species_data(
                fasta_file, species_regex)
            for species_id, seqs in species_to_sequences.items():
                if species_id not in elementos_transponibles[
                        'retrotransposones']:
                    generator = generator_trans
                else:
                    generator = generator_retro

                print(f"........... Especie: {species_id} | reales: {len(seqs)}")
                optimal_total, per_class_limits = find_optimal_augmentation_limit_per_species(
                    generator,
                    fasta_file,
                    species_id,
                    seqs,
                    species_to_class_counts.get(species_id, Counter()),
                    per_step_batch=int(os.environ.get('SPECIES_BATCH', '500')))
                print(
                    f"KNEE Optimo por especie {species_id}: {optimal_total} sinteticas"
                )
                for class_idx, limit in per_class_limits.items():
                    current_count = class_counts.get(class_idx, 0)
                    optimal_limits_list.append({
                        'class_name':
                        inv_superf_dict[class_idx],
                        'class_idx':
                        class_idx,
                        'original_count':
                        current_count,
                        'synthetic_limit':
                        limit
                    })
        else:
            total_samples = sum(class_counts.values())
            for class_idx, current_count in class_counts.items():
                if class_idx not in elementos_transponibles[
                        'retrotransposones']:
                    generator = generator_trans
                else:
                    generator = generator_retro

                class_percentage = (current_count / total_samples
                                    ) * 100 if total_samples > 0 else 0

                if current_count < max_count and class_percentage < 5.0:
                    optimal_synth_count = find_optimal_augmentation_limit(
                        generator, fasta_file, class_idx, max_count)

                    optimal_limits_list.append({
                        'class_name':
                        inv_superf_dict[class_idx],
                        'class_idx':
                        class_idx,
                        'original_count':
                        current_count,
                        'synthetic_limit':
                        optimal_synth_count
                    })
                    print(
                        f"\n--- LIMITE OPTIMO para Clase {inv_superf_dict[class_idx]}: {optimal_synth_count} muestras sinteticas ---"
                    )
                elif class_percentage >= 5.0:
                    print(
                        f"\n--- Clase {inv_superf_dict[class_idx]} tiene {class_percentage:.2f}% del total. No se evalua augmentation limit. ---"
                    )
                else:
                    print(
                        f"\n--- Clase {inv_superf_dict[class_idx]} ya es mayoritaria ({current_count}). Saltando evaluacion. ---"
                    )
        
        if optimal_limits_list:
            with open(OUTPUT_CSV, 'w', newline='') as csvfile:
                fieldnames = [
                    'class_name', 'class_idx', 'original_count',
                    'synthetic_limit'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(optimal_limits_list)

            print(f"Limites optimos guardados en: {OUTPUT_CSV}")