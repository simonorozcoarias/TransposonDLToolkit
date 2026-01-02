#!/usr/bin/env python3
"""
Selección de Especies por Categoría para Entrenamiento de Producción

Este script selecciona aleatoriamente N especies de cada categoría taxonómica
(animal, planta, hongo, otro) para crear un dataset multi-reino balanceado.

Características:
- Selección reproducible con seed fijo
- Validación de existencia de datasets
- Cálculo de imbalance ratio agregado
- Múltiples formatos de salida (texto, JSON, stdout)

Uso:
    python select_species_by_category.py \
        --animals 15 --plants 10 --fungi 10 --other 5 \
        --csv results/species_gc_data_v2.csv \
        --datasets-dir ~/inpactor3/auto_detection/phase2/datasets \
        --seed 42 \
        --output selected_species.txt \
        --json-output species_metadata.json
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


# Mapeo de categorías en español a inglés
CATEGORY_MAP = {
    'animal': 'animal',
    'planta': 'plant',
    'hongo': 'fungus',
    'otro': 'other'
}

# Categorías inversas para facilitar búsqueda
CATEGORY_TO_SPANISH = {v: k for k, v in CATEGORY_MAP.items()}


def parse_species_csv(csv_path: Path) -> Dict[str, List[str]]:
    """
    Parse el CSV de especies y organiza por categoría

    Args:
        csv_path: Ruta al archivo species_gc_data_v2.csv

    Returns:
        Dict con categorías como keys y listas de especies como values

    Note:
        Normaliza nombres de especies reemplazando espacios con guiones bajos
        para consistencia con nombres de directorios y all_species_coverage.csv
    """
    species_by_category = defaultdict(list)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            species_name = row.get('species', '').strip()
            organism_type = row.get('organism_type', '').strip()
            source = row.get('source', '').strip()

            # Skip especies fallidas o sin datos válidos
            if not species_name or source == 'failed':
                continue

            # NORMALIZACIÓN: Convertir espacios a guiones bajos
            # species_gc_data_v2.csv usa espacios ("Aedes aegypti")
            # pero directorios y all_species_coverage.csv usan guiones bajos ("Aedes_aegypti")
            species_name_normalized = species_name.replace(' ', '_')

            # Agregar a la categoría correspondiente
            if organism_type in CATEGORY_MAP:
                english_category = CATEGORY_MAP[organism_type]
                species_by_category[english_category].append(species_name_normalized)

    return dict(species_by_category)


def validate_species_dataset(species_name: str, datasets_dir: Path) -> Tuple[bool, str]:
    """
    Valida que una especie tenga su dataset completo con splits

    Args:
        species_name: Nombre de la especie
        datasets_dir: Directorio base de datasets

    Returns:
        Tupla (is_valid, error_message)
    """
    species_path = datasets_dir / species_name.replace(" ", "_")

    # Verificar que el directorio existe
    if not species_path.exists():
        return False, f"Directory not found: {species_path}"

    if not species_path.is_dir():
        return False, f"Not a directory: {species_path}"

    # Verificar que existen los 3 splits requeridos
    required_splits = ['train', 'val', 'test']
    for split in required_splits:
        split_path = species_path / split
        if not split_path.exists():
            return False, f"Missing '{split}' split"

    return True, "OK"


def select_species_by_category(
    species_by_category: Dict[str, List[str]],
    n_animals: int,
    n_plants: int,
    n_fungi: int,
    n_other: int,
    datasets_dir: Path,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Selecciona N especies de cada categoría con validación

    Args:
        species_by_category: Dict con especies organizadas por categoría
        n_animals: Número de animales a seleccionar
        n_plants: Número de plantas a seleccionar
        n_fungi: Número de hongos a seleccionar
        n_other: Número de otros organismos a seleccionar
        datasets_dir: Directorio donde están los datasets
        seed: Semilla para reproducibilidad

    Returns:
        Dict con especies seleccionadas por categoría
    """
    # Configurar seed para reproducibilidad
    random.seed(seed)

    # Mapeo de categorías a número requerido
    requirements = {
        'animal': n_animals,
        'plant': n_plants,
        'fungus': n_fungi,
        'other': n_other
    }

    selected_species = {}

    for category, n_required in requirements.items():
        available_species = species_by_category.get(category, [])

        # Verificar que hay suficientes especies
        if len(available_species) < n_required:
            print(f"❌ ERROR: Insufficient {category} species", file=sys.stderr)
            print(f"   Required: {n_required}, Available: {len(available_species)}", file=sys.stderr)
            sys.exit(1)

        print(f"\n{category.upper()}: Selecting {n_required} from {len(available_species)} available")

        # Crear una copia para no modificar el original
        candidates = available_species.copy()
        random.shuffle(candidates)

        # Seleccionar y validar especies
        selected = []
        validated_count = 0
        skipped_count = 0

        for species_name in candidates:
            if len(selected) >= n_required:
                break

            # Validar que la especie tiene dataset completo
            is_valid, error_msg = validate_species_dataset(species_name, datasets_dir)

            if is_valid:
                selected.append(species_name)
                validated_count += 1
                print(f"  ✓ {species_name}")
            else:
                skipped_count += 1
                print(f"  ✗ {species_name}: {error_msg}")

        # Verificar que se logró seleccionar el número requerido
        if len(selected) < n_required:
            print(f"❌ ERROR: Could not find {n_required} valid {category} species", file=sys.stderr)
            print(f"   Found: {len(selected)}, Skipped: {skipped_count}", file=sys.stderr)
            sys.exit(1)

        selected_species[category] = selected
        print(f"✅ {category.capitalize()}: Selected {len(selected)} valid species")

    return selected_species


def calculate_imbalance_ratio(
    species_list: List[str],
    coverage_csv_path: Path
) -> float:
    """
    Calcula el imbalance ratio agregado para las especies seleccionadas

    Args:
        species_list: Lista de nombres de especies (con guiones bajos)
        coverage_csv_path: Ruta a all_species_coverage.csv

    Returns:
        Imbalance ratio (background_bases / te_bases)

    Note:
        Asume que species_list usa formato con guiones bajos (Aedes_aegypti)
        que coincide con el formato en all_species_coverage.csv
    """
    if not coverage_csv_path.exists():
        print(f"⚠️  WARNING: Coverage CSV not found: {coverage_csv_path}", file=sys.stderr)
        print(f"   Cannot calculate imbalance ratio", file=sys.stderr)
        return None

    total_te_bases = 0
    total_background_bases = 0
    found_species = set()

    try:
        with open(coverage_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # all_species_coverage.csv usa guiones bajos (Aedes_aegypti)
                species_name = row['species'].strip()

                if species_name in species_list:
                    total_te_bases += int(row['total_te_bases'])
                    total_background_bases += int(row['total_background_bases'])
                    found_species.add(species_name)

        # Reportar especies no encontradas
        missing_species = set(species_list) - found_species
        if missing_species:
            print(f"⚠️  WARNING: {len(missing_species)} species not found in coverage CSV", file=sys.stderr)
            for species in sorted(missing_species):
                print(f"   - {species}", file=sys.stderr)

        # Calcular ratio
        if total_te_bases == 0:
            print(f"❌ ERROR: Total TE bases is zero", file=sys.stderr)
            return None

        ratio = total_background_bases / total_te_bases

        print(f"\n{'='*80}")
        print(f"IMBALANCE RATIO CALCULATION")
        print(f"{'='*80}")
        print(f"Species with coverage data: {len(found_species)}/{len(species_list)}")
        print(f"Total background bases: {total_background_bases:,}")
        print(f"Total TE bases: {total_te_bases:,}")
        print(f"Imbalance ratio: {ratio:.4f}")
        print(f"{'='*80}")

        return ratio

    except Exception as e:
        print(f"❌ ERROR calculating imbalance ratio: {e}", file=sys.stderr)
        return None


def save_text_output(selected_species: Dict[str, List[str]], output_path: Path):
    """Guarda lista de especies en formato texto (una por línea)"""
    # Combinar todas las especies en una lista plana
    all_species = []
    for category in ['animal', 'plant', 'fungus', 'other']:
        all_species.extend(selected_species.get(category, []))

    with open(output_path, 'w', encoding='utf-8') as f:
        for species in all_species:
            f.write(f"{species}\n")

    print(f"\n✅ Species list saved to: {output_path}")


def save_json_output(
    selected_species: Dict[str, List[str]],
    species_by_category: Dict[str, List[str]],
    imbalance_ratio: float,
    output_path: Path
):
    """Guarda metadata completa en formato JSON"""
    # Calcular estadísticas
    total_selected = sum(len(species) for species in selected_species.values())

    metadata = {
        "total_species": total_selected,
        "selection_seed": 42,
        "categories": {
            category: {
                "selected": len(species_list),
                "available": len(species_by_category.get(category, [])),
                "species": sorted(species_list)
            }
            for category, species_list in selected_species.items()
        },
        "imbalance_ratio": imbalance_ratio if imbalance_ratio else "N/A"
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ Metadata saved to: {output_path}")


def print_stdout_summary(selected_species: Dict[str, List[str]], imbalance_ratio: float):
    """Imprime resumen para captura por shell script"""
    # Imprimir imbalance ratio en formato machine-readable
    if imbalance_ratio:
        print(f"IMBALANCE_RATIO={imbalance_ratio:.4f}")

    # Resumen de selección
    total = sum(len(species) for species in selected_species.values())
    print(f"\nSELECTION_SUMMARY:")
    for category in ['animal', 'plant', 'fungus', 'other']:
        n_selected = len(selected_species.get(category, []))
        print(f"  {category}: {n_selected}")
    print(f"  total: {total}")


def main():
    parser = argparse.ArgumentParser(
        description='Selección de especies por categoría para entrenamiento multi-reino',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Número de especies por categoría
    parser.add_argument('--animals', type=int, default=15,
                        help='Number of animal species to select (default: 15)')
    parser.add_argument('--plants', type=int, default=10,
                        help='Number of plant species to select (default: 10)')
    parser.add_argument('--fungi', type=int, default=10,
                        help='Number of fungus species to select (default: 10)')
    parser.add_argument('--other', type=int, default=5,
                        help='Number of other species to select (default: 5)')

    # Archivos de entrada
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to species_gc_data_v2.csv')
    parser.add_argument('--datasets-dir', type=str, required=True,
                        help='Path to datasets directory')
    parser.add_argument('--coverage-csv', type=str, default=None,
                        help='Path to all_species_coverage.csv for imbalance ratio calculation')

    # Configuración
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    # Archivos de salida
    parser.add_argument('--output', type=str, required=True,
                        help='Output text file with selected species (one per line)')
    parser.add_argument('--json-output', type=str, default=None,
                        help='Optional JSON output file with metadata')

    args = parser.parse_args()

    # Convertir paths a Path objects
    csv_path = Path(args.csv).expanduser()
    datasets_dir = Path(args.datasets_dir).expanduser()
    output_path = Path(args.output)
    json_output_path = Path(args.json_output) if args.json_output else None

    # Verificar archivos de entrada
    if not csv_path.exists():
        print(f"❌ ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if not datasets_dir.exists():
        print(f"❌ ERROR: Datasets directory not found: {datasets_dir}", file=sys.stderr)
        sys.exit(1)

    print("="*80)
    print("SPECIES SELECTION BY CATEGORY")
    print("="*80)
    print(f"Input CSV: {csv_path}")
    print(f"Datasets dir: {datasets_dir}")
    print(f"Selection targets:")
    print(f"  - Animals: {args.animals}")
    print(f"  - Plants: {args.plants}")
    print(f"  - Fungi: {args.fungi}")
    print(f"  - Other: {args.other}")
    print(f"  - TOTAL: {args.animals + args.plants + args.fungi + args.other}")
    print(f"Random seed: {args.seed}")
    print("="*80)

    # 1. Parse CSV y organizar por categorías
    print("\nStep 1: Parsing species CSV...")
    species_by_category = parse_species_csv(csv_path)

    for category in ['animal', 'plant', 'fungus', 'other']:
        n_available = len(species_by_category.get(category, []))
        print(f"  {category:10s}: {n_available:4d} species available")

    # 2. Seleccionar especies con validación
    print("\nStep 2: Selecting and validating species...")
    selected_species = select_species_by_category(
        species_by_category,
        args.animals,
        args.plants,
        args.fungi,
        args.other,
        datasets_dir,
        args.seed
    )

    # 3. Calcular imbalance ratio
    imbalance_ratio = None
    if args.coverage_csv:
        coverage_csv_path = Path(args.coverage_csv).expanduser()
        print("\nStep 3: Calculating imbalance ratio...")

        # Crear lista plana de todas las especies seleccionadas
        all_selected = []
        for species_list in selected_species.values():
            all_selected.extend(species_list)

        imbalance_ratio = calculate_imbalance_ratio(all_selected, coverage_csv_path)
    else:
        print("\nStep 3: Skipping imbalance ratio calculation (no coverage CSV provided)")

    # 4. Guardar salidas
    print("\nStep 4: Saving outputs...")

    # Texto: lista de especies
    save_text_output(selected_species, output_path)

    # JSON: metadata completa (opcional)
    if json_output_path:
        save_json_output(selected_species, species_by_category, imbalance_ratio, json_output_path)

    # STDOUT: valores machine-readable para bash
    print("\n" + "="*80)
    print_stdout_summary(selected_species, imbalance_ratio)
    print("="*80)

    print("\n✅ Species selection completed successfully")


if __name__ == '__main__':
    main()
