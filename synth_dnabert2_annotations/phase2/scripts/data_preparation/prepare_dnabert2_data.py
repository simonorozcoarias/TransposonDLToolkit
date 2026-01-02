#!/usr/bin/env python3
"""
Preparación de datos para DNABERT-2: Token Classification Task
Convierte cromosomas sintéticos FASTA + GFF3 a formato de entrenamiento

Estrategia:
1. Sliding windows sobre cromosomas (2048 bp con 50% overlap)
2. Tokenización con DNABERT-2 BPE tokenizer
3. Etiquetas binarias a nivel de token (TE vs background)
4. Genera splits train/val/test (80/10/10) automáticamente
5. Exporta a formato Hugging Face Dataset optimizado
"""

import os
import gzip
from pathlib import Path
from collections import defaultdict
import numpy as np
from Bio import SeqIO
import torch
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset

class TEAnnotation:
    """Representa una anotación de TE"""
    def __init__(self, start, end, family, strand):
        self.start = start
        self.end = end
        self.family = family
        self.strand = strand
    
    def overlaps(self, pos):
        """Verifica si una posición está dentro del TE"""
        return self.start <= pos <= self.end

def parse_gff3_indexed(gff_file):
    """
    Parsea GFF3 y retorna diccionario indexado por seqid
    """
    annotations = defaultdict(list)
    
    open_func = gzip.open if gff_file.endswith('.gz') else open
    
    with open_func(gff_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            seqid, source, feature_type, start, end, score, strand, phase, attributes = fields
            
            # Parsear familia de atributos
            family = 'unknown'
            for attr in attributes.split(';'):
                if attr.startswith('family='):
                    family = attr.split('=')[1]
                    break
            
            te_ann = TEAnnotation(
                start=int(start) - 1,  # GFF3 es 1-based, convertir a 0-based
                end=int(end),
                family=family,
                strand=strand
            )
            annotations[seqid].append(te_ann)
    
    return annotations

def create_binary_labels(sequence, annotations):
    """
    Crea etiquetas binarias para cada posición nucleótido

    Esquema simplificado:
    - 0: Background (no TE)
    - 1: TE (cualquier elemento transponible)

    Retorna array de etiquetas binarias
    """
    seq_len = len(sequence)
    labels = [0] * seq_len  # Por defecto: background

    # Marcar todas las posiciones que están dentro de un TE
    for ann in annotations:
        for pos in range(ann.start, min(ann.end, seq_len)):
            labels[pos] = 1  # TE

    return labels

def generate_sliding_windows(sequence, window_size=2048, stride=1024):
    """
    Genera ventanas deslizantes sobre la secuencia

    Args:
        sequence: str, secuencia de ADN
        window_size: int, tamaño de ventana en nucleótidos (default: 2048)
        stride: int, desplazamiento entre ventanas (default: 1024, 50% overlap)

    Yields:
        tuple (start_pos, window_seq)
    """
    seq_len = len(sequence)
    
    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        yield (start, sequence[start:end])
    
    # Última ventana si no llegamos al final
    if seq_len % stride != 0:
        start = max(0, seq_len - window_size)
        yield (start, sequence[start:seq_len])

def align_labels_to_tokens(labels, offset_mapping):
    """
    Alinea etiquetas a nivel de nucleótido con tokens BPE

    Args:
        labels: lista de etiquetas por nucleótido (0 o 1)
        offset_mapping: lista de tuplas (start, end) por token

    Returns:
        Lista de etiquetas alineadas con tokens
        Usa -100 para tokens especiales (CLS, SEP, PAD)
    """
    token_labels = []

    for start, end in offset_mapping:
        # Tokens especiales (CLS, SEP, PAD) tienen offset (0, 0)
        if start == end == 0:
            token_labels.append(-100)  # Ignorar en loss
        else:
            # Asignar label del primer nucleótido del token
            # Alternativa: mayoría de votos de labels[start:end]
            token_labels.append(labels[start])

    return token_labels

def prepare_dataset(fasta_file, gff_file, output_dir,
                   window_size=2048, stride=1024,
                   model_max_length=512,
                   model_name="zhihan1996/DNABERT-2-117M",
                   max_samples=None,
                   species_name=None,
                   create_splits=True,
                   train_ratio=0.8,
                   val_ratio=0.1,
                   test_ratio=0.1,
                   split_seed=42):
    """
    Prepara dataset completo para entrenamiento con DNABERT-2

    Args:
        fasta_file: Archivo FASTA con secuencias
        gff_file: Archivo GFF3 con anotaciones de TEs
        output_dir: Directorio de salida
        window_size: Tamaño de ventana en nucleótidos (default: 2048)
        stride: Desplazamiento entre ventanas (default: 1024)
        model_max_length: Longitud máxima para tokenizer (default: 512)
        model_name: Modelo DNABERT-2 a usar (default: DNABERT-2-117M)
        max_samples: Límite de samples para testing (default: None)
        species_name: Nombre de la especie (para trazabilidad en datasets combinados)
        create_splits: Si True, crea splits train/val/test automáticamente (default: True)
        train_ratio: Proporción para entrenamiento (default: 0.8)
        val_ratio: Proporción para validación (default: 0.1)
        test_ratio: Proporción para test (default: 0.1)
        split_seed: Semilla para splits reproducibles (default: 42)

    Returns:
        tuple: (samples_list, label_vocab)
    """
    print(f"Procesando: {fasta_file}")
    print(f"Parámetros: window_size={window_size}, stride={stride}")

    # Cargar tokenizer de DNABERT-2
    print(f"\nCargando tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        model_max_length=model_max_length
    )
    print(f"Tokenizer cargado. Max length: {model_max_length}")

    # Parsear anotaciones
    annotations = parse_gff3_indexed(gff_file)
    print(f"Anotaciones cargadas: {len(annotations)} secuencias")

    # Vocabulario binario
    label_vocab = {0: "background", 1: "TE"}

    # Leer secuencias FASTA
    samples = []
    n_samples = 0

    for record in SeqIO.parse(fasta_file, "fasta"):
        seqid = record.id
        sequence = str(record.seq).upper()

        # Obtener anotaciones para esta secuencia
        seq_annotations = annotations.get(seqid, [])

        print(f"  Procesando {seqid}: {len(sequence):,} bp, {len(seq_annotations)} TEs")

        # Generar ventanas deslizantes
        for window_start, window_seq in generate_sliding_windows(
            sequence, window_size, stride
        ):
            # Filtrar anotaciones relevantes para esta ventana
            window_end = window_start + len(window_seq)
            window_annotations = [
                ann for ann in seq_annotations
                if ann.start < window_end and ann.end > window_start
            ]

            # Ajustar coordenadas relativas a la ventana
            adjusted_annotations = []
            for ann in window_annotations:
                adj_ann = TEAnnotation(
                    start=max(0, ann.start - window_start),
                    end=min(len(window_seq), ann.end - window_start),
                    family=ann.family,
                    strand=ann.strand
                )
                adjusted_annotations.append(adj_ann)

            # Crear etiquetas binarias a nivel de nucleótido
            nucleotide_labels = create_binary_labels(
                window_seq, adjusted_annotations
            )

            # Tokenizar con DNABERT-2 BPE
            tokenized = tokenizer(
                window_seq,
                truncation=True,
                max_length=model_max_length,
                return_offsets_mapping=True,
                add_special_tokens=True
            )

            # Alinear labels con tokens usando offset mapping
            token_labels = align_labels_to_tokens(
                nucleotide_labels,
                tokenized['offset_mapping']
            )

            sample = {
                'sequence_id': seqid,
                'window_start': window_start,
                'sequence': window_seq,
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': token_labels,
                'n_tes': len(adjusted_annotations),
                'te_families': [ann.family for ann in adjusted_annotations]
            }

            # Añadir nombre de especie si está disponible
            if species_name:
                sample['species'] = species_name

            samples.append(sample)
            n_samples += 1

            if max_samples and n_samples >= max_samples:
                break

        if max_samples and n_samples >= max_samples:
            break

    print(f"\nTotal samples generados: {len(samples)}")

    # Guardar dataset
    os.makedirs(output_dir, exist_ok=True)

    # 1. Guardar como Hugging Face Dataset (formato principal)
    hf_dict = {
        'sequence_id': [s['sequence_id'] for s in samples],
        'window_start': [s['window_start'] for s in samples],
        'input_ids': [s['input_ids'] for s in samples],
        'attention_mask': [s['attention_mask'] for s in samples],
        'labels': [s['labels'] for s in samples],
        'n_tes': [s['n_tes'] for s in samples],
    }

    # Añadir campo species si está presente
    if species_name:
        hf_dict['species'] = [s['species'] for s in samples]

    hf_dataset = HFDataset.from_dict(hf_dict)
    hf_output = os.path.join(output_dir, 'hf_dataset')
    hf_dataset.save_to_disk(hf_output)
    print(f"[HF Dataset] Guardado: {hf_output}")

    # 2. Guardar vocabulario
    vocab_file = os.path.join(output_dir, 'label_vocabulary.json')
    with open(vocab_file, 'w') as f:
        json.dump(label_vocab, f, indent=2)

    # 3. Guardar estadísticas
    stats = {
        'n_samples': len(samples),
        'n_labels': len(label_vocab),
        'window_size': window_size,
        'stride': stride,
        'model_max_length': model_max_length,
        'model_name': model_name,
        'label_scheme': 'binary',
        'total_tes': sum(s['n_tes'] for s in samples)
    }
    stats_file = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nArchivos generados:")
    print(f"  - {hf_output}/")
    print(f"  - {vocab_file}")
    print(f"  - {stats_file}")

    # 5. Crear splits train/val/test si está habilitado
    if create_splits:
        print(f"\n{'='*80}")
        print("CREANDO SPLITS TRAIN/VAL/TEST")
        print(f"{'='*80}")
        print(f"Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        print(f"Seed: {split_seed}")

        # Validar ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            print(f"⚠️  WARNING: Los ratios no suman 1.0 (suma: {total_ratio})")
            print(f"   Normalizando ratios...")
            train_ratio = train_ratio / total_ratio
            val_ratio = val_ratio / total_ratio
            test_ratio = test_ratio / total_ratio

        # Split train/temp
        print(f"\nDividiendo dataset ({len(hf_dataset):,} muestras)...")
        splits = hf_dataset.train_test_split(
            train_size=train_ratio,
            seed=split_seed
        )
        train_dataset = splits['train']
        temp_dataset = splits['test']

        # Split val/test
        val_size = val_ratio / (val_ratio + test_ratio)
        splits2 = temp_dataset.train_test_split(
            train_size=val_size,
            seed=split_seed
        )
        val_dataset = splits2['train']
        test_dataset = splits2['test']

        print(f"✓ Splits creados:")
        print(f"  Train:      {len(train_dataset):8,} muestras ({len(train_dataset)/len(hf_dataset)*100:.1f}%)")
        print(f"  Validation: {len(val_dataset):8,} muestras ({len(val_dataset)/len(hf_dataset)*100:.1f}%)")
        print(f"  Test:       {len(test_dataset):8,} muestras ({len(test_dataset)/len(hf_dataset)*100:.1f}%)")

        # Guardar cada split
        print(f"\nGuardando splits...")
        train_path = os.path.join(output_dir, 'train')
        val_path = os.path.join(output_dir, 'val')
        test_path = os.path.join(output_dir, 'test')

        train_dataset.save_to_disk(train_path)
        print(f"  ✓ Train:      {train_path}")

        val_dataset.save_to_disk(val_path)
        print(f"  ✓ Validation: {val_path}")

        test_dataset.save_to_disk(test_path)
        print(f"  ✓ Test:       {test_path}")

        # Actualizar estadísticas con info de splits
        stats['splits'] = {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'seed': split_seed
        }

        # Guardar estadísticas actualizadas
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n{'='*80}")
        print("✓ SPLITS CREADOS EXITOSAMENTE")
        print(f"{'='*80}")

    return samples, label_vocab

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Prepara datos de TEs para entrenamiento con DNABERT-2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Argumentos requeridos
    parser.add_argument('fasta', help='Archivo FASTA con secuencias genómicas')
    parser.add_argument('gff3', help='Archivo GFF3 con anotaciones de TEs')
    parser.add_argument('output_dir', help='Directorio de salida para dataset')

    # Parámetros de ventanas
    parser.add_argument('--window-size', type=int, default=2048,
                       help='Tamaño de ventana en nucleótidos')
    parser.add_argument('--stride', type=int, default=1024,
                       help='Desplazamiento entre ventanas (overlap = window_size - stride)')

    # Parámetros del modelo
    parser.add_argument('--model-max-length', type=int, default=512,
                       help='Longitud máxima para tokenizer DNABERT-2')
    parser.add_argument('--model-name', type=str, default='zhihan1996/DNABERT-2-117M',
                       help='Nombre del modelo DNABERT-2 en HuggingFace')

    # Opciones de procesamiento
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Límite de samples para testing (None = todos)')
    parser.add_argument('--species', type=str, default=None,
                       help='Nombre de la especie (para trazabilidad en datasets combinados)')

    # Opciones de splits
    parser.add_argument('--no-splits', action='store_true',
                       help='No crear splits train/val/test automáticamente')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Proporción para entrenamiento')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Proporción para validación')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Proporción para test')
    parser.add_argument('--split-seed', type=int, default=42,
                       help='Semilla para splits reproducibles')

    args = parser.parse_args()

    print("=" * 80)
    print("PREPARACIÓN DE DATOS PARA DNABERT-2")
    print("=" * 80)
    print(f"\nInputs:")
    print(f"  FASTA: {args.fasta}")
    print(f"  GFF3:  {args.gff3}")
    print(f"  Output: {args.output_dir}")
    print(f"\nConfiguración:")
    print(f"  Window size: {args.window_size} bp")
    print(f"  Stride: {args.stride} bp (overlap: {args.window_size - args.stride} bp)")
    print(f"  Model: {args.model_name}")
    print(f"  Max length: {args.model_max_length} tokens")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}")
    print()

    samples, vocab = prepare_dataset(
        fasta_file=args.fasta,
        gff_file=args.gff3,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        model_max_length=args.model_max_length,
        model_name=args.model_name,
        max_samples=args.max_samples,
        species_name=args.species,
        create_splits=not args.no_splits,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed
    )

    print("\n" + "=" * 80)
    print("✓ PREPARACIÓN COMPLETADA")
    print("=" * 80)
