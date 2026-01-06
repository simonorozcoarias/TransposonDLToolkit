#!/usr/bin/env python3
"""
Script para combinar el dataset sintético (necesita curación) 
con el archivo original de TEs (no necesita curación)
"""
from Bio import SeqIO
import argparse
import random
import os
from pathlib import Path


def parse_synthetic_header(header):
    """
    Parsea el header del dataset sintético
    
    - Caso1: >Caso1_full_AluSg_15#CLASSI/SINE/ALU_4970_310_Hominoidea
    - Caso2: >Caso2_full_ID_0_len_species
    - Caso3: >Caso3_full_ID_pos_len_species
    - Caso4: >Caso4_microsat_0_len_species
    """
    header = header.replace('>', '').strip()
    parts = header.split('_', 2)
    
    if len(parts) < 3:
        return {'case': parts[0] if parts else 'unknown', 'te_id': header}
    
    case_type = parts[0]
    frag_status = parts[1]
    rest = parts[2]
    
    te_info = {
        'case': case_type,
        'fragmented': frag_status == 'frag',
        'is_microsat': frag_status == 'microsat'
    }
    if '#' in rest:
        te_id_part = rest.split('#')[0]
        info = rest.split('#', 1)[1] if '#' in rest else ""
        te_info['info'] = info
    else:
        te_id_part = rest
        te_info['info'] = ""
    te_parts = te_id_part.split('_')
    
    te_id = te_id_part
    position = None
    length = None
    species = None
    
    numeric_parts = []
    for i, part in enumerate(te_parts):
        if part.isdigit():
            numeric_parts.append((i, int(part)))
    
    if len(numeric_parts) >= 2:
        pos_idx, position = numeric_parts[-2]
        len_idx, length = numeric_parts[-1]
        te_id = '_'.join(te_parts[:pos_idx])
        if len_idx + 1 < len(te_parts):
            species = '_'.join(te_parts[len_idx + 1:])
    elif len(numeric_parts) == 1:
        len_idx, length = numeric_parts[0]
        te_id = '_'.join(te_parts[:len_idx])
        if len_idx + 1 < len(te_parts):
            species = '_'.join(te_parts[len_idx + 1:])
    else:
        te_id = te_id_part
    
    te_info['te_id'] = te_id
    if position is not None:
        te_info['position'] = position
    if length is not None:
        te_info['length'] = length
    if species:
        te_info['species'] = species
    
    return te_info

def combine_datasets(synthetic_file, original_file, output_file, labels_file=None, 
                     max_original=None, balance=True):
    """
    Combina el dataset sintético con el original
    """
    synthetic_sequences = []
    synthetic_labels = []
    
    with open(synthetic_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            synthetic_sequences.append(record)
            # header
            te_info = parse_synthetic_header(record.description)
            if te_info:
                # determinar el caso
                case_type = te_info['case']
                if case_type in ['Caso4', 'Caso5']:
                    label_type = "no_te"
                else:
                    label_type = "needs_curation"
                    
                label = f"{label_type}\t{case_type}\t{te_info['te_id']}\t{te_info.get('position', 'N/A')}\t{te_info.get('length', 'N/A')}"
            else:
                label = f"needs_curation\t{record.description}"
            synthetic_labels.append(label)
    
    original_sequences = []
    original_labels = []
    
    with open(original_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            # header
            if "#" in record.id:
                record.id = record.id.replace("#", "_")
            if "#" in record.description:
                record.description = record.description.replace("#", "_")
                
            original_sequences.append(record)
            original_labels.append(f"no_curation_needed\t{record.description}")
    
    if balance and len(original_sequences) > len(synthetic_sequences):
        print(f"Balanceado: seleccionando {len(synthetic_sequences)} secuencias del original")
        selected_indices = random.sample(range(len(original_sequences)), len(synthetic_sequences))
        original_sequences = [original_sequences[i] for i in selected_indices]
        original_labels = [original_labels[i] for i in selected_indices]
    elif max_original:
        print(f"Limitando dataset original a {max_original} secuencias")
        if len(original_sequences) > max_original:
            selected_indices = random.sample(range(len(original_sequences)), max_original)
            original_sequences = [original_sequences[i] for i in selected_indices]
            original_labels = [original_labels[i] for i in selected_indices]
    
    all_sequences = synthetic_sequences + original_sequences
    all_labels = synthetic_labels + original_labels
    
    # Mezclar secuencias y etiquetas
    combined = list(zip(all_sequences, all_labels))
    random.shuffle(combined)
    all_sequences, all_labels = zip(*combined)
    
    print(f"\nEscribiendo archivo combinado: {output_file}")
    with open(output_file, "w") as f:
        for record in all_sequences:
            SeqIO.write(record, f, "fasta")
    
    print(f"\nEscribiendo etiquetas: {labels_file}")
    with open(labels_file, "w") as f:
        f.write("sequence_id\tlabel\tcase_type\tte_id\tposition\tlength\n")
        for i, (record, label) in enumerate(zip(all_sequences, all_labels)):
            seq_id = record.id if record.id else f"seq_{i}"
            f.write(f"{seq_id}\t{label}\n")
    
    print(f"\nResumen del dataset combinado:")
    print(f"   Total de secuencias: {len(all_sequences)}")
    print(f"   - Necesitan curación: {len(synthetic_sequences)}")
    print(f"   - No necesitan curación: {len(original_sequences)}")
    print(f"   Proporción: {len(synthetic_sequences)}/{len(original_sequences)} = {len(synthetic_sequences)/len(original_sequences):.2f}")
