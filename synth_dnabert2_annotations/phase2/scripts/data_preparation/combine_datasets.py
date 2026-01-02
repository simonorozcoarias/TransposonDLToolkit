#!/usr/bin/env python3
"""
Combina splits (train/val/test) de múltiples especies en datasets combinados
Aplica shuffle para mezclar especies y evitar sesgo por orden

NUEVA ESTRATEGIA (Split-Then-Combine):
- Cada especie ya tiene sus propios splits train/val/test
- Este script combina por separado todos los train, todos los val, todos los test
- Aplica shuffle con seed fijo para reproducibilidad

Uso:
    python combine_datasets.py <datasets_dir> <output_dir> [--no-shuffle] [--seed SEED]
"""

import os
import sys
import json
import time
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets, Dataset as HFDataset
from collections import defaultdict


def calculate_imbalance_ratio_from_csv(coverage_csv_path, species_list):
    """
    Calcula el imbalance_ratio agregado para una lista de especies

    Args:
        coverage_csv_path: Ruta al all_species_coverage.csv
        species_list: Lista de nombres de especies

    Returns:
        float: imbalance_ratio (background/TE) o None si error
    """
    import csv

    total_te_bases = 0
    total_background_bases = 0
    found_species = set()

    try:
        with open(coverage_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                species_name = row['species']
                if species_name in species_list:
                    total_te_bases += int(row['total_te_bases'])
                    total_background_bases += int(row['total_background_bases'])
                    found_species.add(species_name)

        # Verificar que se encontraron todas las especies
        missing = set(species_list) - found_species
        if missing:
            print(f"⚠️  Species not found in CSV: {missing}")

        if total_te_bases == 0:
            print(f"❌ ERROR: Total TE bases is zero")
            return None

        ratio = total_background_bases / total_te_bases
        return ratio

    except Exception as e:
        print(f"❌ ERROR calculating imbalance ratio: {e}")
        return None


def find_species_with_splits(base_dir):
    """
    Encuentra todas las especies que tienen splits train/val/test

    Args:
        base_dir: Directorio base que contiene subdirectorios de especies

    Returns:
        dict: {species_name: species_dir_path}
    """
    species_dirs = {}
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return species_dirs

    # Buscar en cada subdirectorio
    for species_dir in base_path.iterdir():
        if not species_dir.is_dir():
            continue

        # Verificar que existan los 3 splits
        train_path = species_dir / "train"
        val_path = species_dir / "val"
        test_path = species_dir / "test"

        if train_path.exists() and val_path.exists() and test_path.exists():
            species_name = species_dir.name
            species_dirs[species_name] = str(species_dir)
            print(f"  Found species with splits: {species_name}")

    return species_dirs

def combine_single_split(species_dirs, split_name, output_dir, shuffle=True, seed=42):
    """
    Combina un split específico (train/val/test) de todas las especies

    Args:
        species_dirs: Dict de {species_name: species_dir_path}
        split_name: Nombre del split ('train', 'val', o 'test')
        output_dir: Directorio de salida
        shuffle: Si True, mezcla las muestras (default: True)
        seed: Semilla para shuffle reproducible (default: 42)

    Returns:
        Dataset combinado
    """
    print(f"\n{'='*80}")
    print(f"COMBINING {split_name.upper()} SPLIT")
    print(f"{'='*80}")

    datasets_list = []
    species_stats = {}

    # Cargar el split de cada especie
    for species_name in sorted(species_dirs.keys()):
        species_dir = species_dirs[species_name]
        split_path = os.path.join(species_dir, split_name)

        if not os.path.exists(split_path):
            print(f"  ⚠️  {species_name}/{split_name}: No encontrado, omitiendo...")
            continue

        try:
            dataset = load_from_disk(split_path)
            n_samples = len(dataset)
            datasets_list.append(dataset)
            species_stats[species_name] = {
                'n_samples': n_samples,
                'n_tes': sum(dataset['n_tes']) if 'n_tes' in dataset.column_names else 0
            }
            print(f"  ✓ {species_name:30s}: {n_samples:8,} samples")
        except Exception as e:
            print(f"  ✗ {species_name}: Error al cargar - {e}")
            continue

    if not datasets_list:
        print(f"\n❌ ERROR: No se encontraron datasets para el split '{split_name}'")
        return None

    # Combinar todos los datasets
    print(f"\n  Concatenando {len(datasets_list)} datasets...")
    combined = concatenate_datasets(datasets_list)
    print(f"  ✓ Dataset combinado: {len(combined):,} muestras")

    # ESTRATEGIA CRÍTICA para evitar degradación por especie:
    # 1. Crear índices shuffleados (rápido, solo ~100MB de memoria)
    # 2. Guardar con .select() aplicando los índices shuffleados
    # Esto evita materializar todo el dataset shuffleado en memoria

    output_path = os.path.join(output_dir, split_name)

    if shuffle:
        print(f"\n  🔀 Generando índices shuffleados (seed={seed})...")
        print(f"  ℹ️  Esto previene degradación por cambio de especie sin materializar dataset completo")
        sys.stdout.flush()

        import numpy as np
        # Generar índices shuffleados (muy rápido y bajo uso de memoria)
        t0 = time.time()
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(combined))
        print(f"  ✓ Índices generados en {(time.time()-t0):.1f}s")

        # Aplicar índices shuffleados con .select()
        print(f"  🔄 Aplicando shuffle mediante .select() con índices...")
        sys.stdout.flush()
        t0 = time.time()
        combined = combined.select(indices)
        print(f"  ✓ Shuffle aplicado en {(time.time()-t0):.1f}s")

    print(f"\n  💾 Guardando dataset combinado: {output_path}")
    sys.stdout.flush()

    print(f"\n  Configuración de guardado:")
    print(f"    - max_shard_size: 2GB por shard")
    print(f"    - num_proc: 16 procesos paralelos")
    print(f"    - Total muestras: {len(combined):,}")
    print(f"    - Orden: {'Shuffleado (especies mezcladas)' if shuffle else 'Por especie (alfabético)'}")
    sys.stdout.flush()

    t0 = time.time()
    combined.save_to_disk(
        output_path,
        max_shard_size="2GB",
        num_proc=16
    )
    t1 = time.time()
    elapsed = t1 - t0
    speed = len(combined) / elapsed if elapsed > 0 else 0
    print(f"  ✓ Guardado en {elapsed:.1f}s ({speed:,.0f} samples/s)")

    # Estadísticas del split
    split_stats = {
        'n_species': len(species_stats),
        'total_samples': len(combined),
        'shuffled': shuffle,
        'seed': seed if shuffle else None,
        'species_breakdown': species_stats
    }

    # NOTA: No calculamos samples_per_species porque requiere iterar sobre 26M samples
    # y es muy lento. La información por especie está en species_breakdown.

    # Guardar estadísticas del split
    stats_file = os.path.join(output_dir, f'{split_name}_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(split_stats, f, indent=2)
    print(f"  ✓ Estadísticas: {stats_file}")

    return combined


def combine_datasets(datasets_dir, output_dir, shuffle=True, seed=42, coverage_csv=None):
    """
    Combina splits de múltiples especies en datasets combinados

    NUEVA ESTRATEGIA (Split-Then-Combine):
    1. Encuentra especies con splits train/val/test
    2. Combina cada split por separado (train, val, test)
    3. Aplica shuffle a cada split para mezclar especies
    4. Guarda splits combinados
    5. Calcula imbalance_ratio agregado si se proporciona coverage_csv

    Args:
        datasets_dir: Directorio con subdirectorios de especies (cada uno con train/val/test/)
        output_dir: Directorio de salida para splits combinados
        shuffle: Si True, mezcla las muestras de cada split (default: True)
        seed: Semilla para shuffle reproducible (default: 42)
        coverage_csv: Ruta al CSV con estadísticas de cobertura (opcional)
    """
    print("="*80)
    print("COMBINING SPECIES SPLITS (Split-Then-Combine Strategy)")
    print("="*80)
    print(f"\nDatasets directory: {datasets_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Shuffle enabled:    {shuffle}")
    if shuffle:
        print(f"Random seed:        {seed}")
    print()

    # Encontrar especies con splits
    print("Buscando especies con splits train/val/test...")
    species_dirs = find_species_with_splits(datasets_dir)

    if not species_dirs:
        print("\n❌ ERROR: No se encontraron especies con splits train/val/test")
        print("\nAsegúrate de que:")
        print("  1. Cada especie tiene directorios train/, val/, test/")
        print("  2. Los directorios contienen datasets HF válidos")
        sys.exit(1)

    print(f"\n✓ Total especies encontradas con splits: {len(species_dirs)}")

    # Calculate imbalance ratio if coverage CSV provided
    imbalance_ratio = None
    if coverage_csv and os.path.exists(coverage_csv):
        print("\n" + "="*80)
        print("CALCULATING IMBALANCE RATIO")
        print("="*80)
        species_names = list(species_dirs.keys())
        imbalance_ratio = calculate_imbalance_ratio_from_csv(coverage_csv, species_names)

        if imbalance_ratio is not None:
            print(f"✅ Imbalance ratio (background/TE): {imbalance_ratio:.4f}")
            # Print in machine-readable format for bash capture
            print(f"IMBALANCE_RATIO={imbalance_ratio:.4f}")
        else:
            print("⚠️  Could not calculate imbalance ratio")
    elif coverage_csv:
        print(f"\n⚠️  WARNING: Coverage CSV file not found: {coverage_csv}")

    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # Combinar cada split por separado
    combined_splits = {}
    for split_name in ['train', 'val', 'test']:
        combined = combine_single_split(
            species_dirs=species_dirs,
            split_name=split_name,
            output_dir=output_dir,
            shuffle=shuffle,
            seed=seed
        )
        if combined is not None:
            combined_splits[split_name] = combined

    # Verificar que se combinaron todos los splits
    if len(combined_splits) != 3:
        print(f"\n⚠️  WARNING: Solo se combinaron {len(combined_splits)}/3 splits")
        if len(combined_splits) == 0:
            print("❌ ERROR: No se pudo combinar ningún split")
            sys.exit(1)

    # Guardar estadísticas globales
    print(f"\n{'='*80}")
    print("RESUMEN FINAL")
    print(f"{'='*80}")

    global_stats = {
        'n_species': len(species_dirs),
        'shuffle': shuffle,
        'seed': seed if shuffle else None,
        'imbalance_ratio': imbalance_ratio,
        'splits': {}
    }

    for split_name, dataset in combined_splits.items():
        global_stats['splits'][split_name] = {
            'n_samples': len(dataset),
            'path': os.path.join(output_dir, split_name)
        }
        print(f"\n{split_name.upper()}:")
        print(f"  Samples: {len(dataset):,}")
        print(f"  Path:    {os.path.join(output_dir, split_name)}/")
        print(f"  (Distribución por especie disponible en {split_name}_stats.json)")

    # Guardar estadísticas globales
    stats_file = os.path.join(output_dir, 'combined_splits_info.json')
    with open(stats_file, 'w') as f:
        json.dump(global_stats, f, indent=2)

    print(f"\n{'='*80}")
    print("✅ COMBINACIÓN DE SPLITS COMPLETADA EXITOSAMENTE")
    print(f"{'='*80}")
    print(f"\nEstadísticas guardadas en: {stats_file}")
    print(f"\nPara usar en entrenamiento:")
    print(f"  from datasets import load_from_disk")
    print(f"  train = load_from_disk('{os.path.join(output_dir, 'train')}')")
    print(f"  val = load_from_disk('{os.path.join(output_dir, 'val')}')")
    print(f"  test = load_from_disk('{os.path.join(output_dir, 'test')}')")
    print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Combina splits de múltiples especies en datasets combinados (Split-Then-Combine)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('datasets_dir',
                       help='Directorio con subdirectorios de especies (cada uno con train/val/test/)')
    parser.add_argument('output_dir',
                       help='Directorio de salida para splits combinados')

    parser.add_argument('--no-shuffle', action='store_true',
                       help='No aplicar shuffle (NO RECOMENDADO - especies quedarán en orden alfabético)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Semilla para shuffle reproducible')
    parser.add_argument('--coverage-csv', default=None,
                       help='Path to all_species_coverage.csv for imbalance ratio calculation')

    args = parser.parse_args()

    combine_datasets(
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        coverage_csv=args.coverage_csv
    )
