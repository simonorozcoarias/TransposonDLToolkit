#!/usr/bin/env python3
"""
Análisis de proporción TE/Background en genomas sintéticos
Autor: Jorge González Gilbaja
TFM - Detección automática de TEs mediante Deep Learning
Fecha: 02/12/2025

Este script analiza los genomas sintéticos generados con TEgenomeSimulator
y calcula la proporción real entre secuencia background y TEs insertados.
"""

import argparse
import os
import sys
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from Bio import SeqIO


def parse_gff_file(gff_file: str) -> List[Dict]:
    """
    Parsear archivo GFF con anotaciones de TEs.

    Args:
        gff_file: Ruta al archivo GFF

    Returns:
        Lista de diccionarios con información de cada TE
    """
    annotations = []

    with open(gff_file, 'r') as f:
        for line in f:
            # Saltar comentarios y headers
            if line.startswith('#'):
                continue

            # Parsear línea GFF
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            seqid, source, feature_type, start, end, score, strand, phase, attributes = fields

            # Todas las anotaciones en el GFF son TEs
            # (lo que no está anotado es background)

            # Parsear atributos
            attr_dict = {}
            for attr in attributes.split(';'):
                if '=' in attr:
                    key, value = attr.split('=', 1)
                    attr_dict[key] = value

            annotation = {
                'seqid': seqid,
                'source': source,
                'type': feature_type,
                'start': int(start),
                'end': int(end),
                'length': int(end) - int(start) + 1,
                'strand': strand,
                'attributes': attr_dict
            }

            annotations.append(annotation)

    return annotations


def get_genome_length(fasta_file: str) -> Dict[str, int]:
    """
    Obtener longitudes de cromosomas del archivo FASTA.

    Args:
        fasta_file: Ruta al archivo FASTA del genoma

    Returns:
        Diccionario con {seqid: length}
    """
    lengths = {}

    for record in SeqIO.parse(fasta_file, "fasta"):
        lengths[record.id] = len(record.seq)

    return lengths


def calculate_te_coverage(annotations: List[Dict], chr_lengths: Dict[str, int]) -> Dict:
    """
    Calcular estadísticas de coverage de TEs.

    Args:
        annotations: Lista de anotaciones de TEs
        chr_lengths: Diccionario con longitudes de cromosomas

    Returns:
        Diccionario con estadísticas
    """
    stats = {
        'total_genome_length': sum(chr_lengths.values()),
        'total_te_bases': 0,
        'total_background_bases': 0,
        'num_te_insertions': len(annotations),
        'te_coverage_percent': 0.0,
        'background_percent': 0.0,
        'te_length_mean': 0.0,
        'te_length_min': 0,
        'te_length_max': 0,
        'chromosomes': {}
    }

    if not annotations:
        stats['total_background_bases'] = stats['total_genome_length']
        stats['background_percent'] = 100.0
        return stats

    # Agrupar por cromosoma
    chr_annotations = defaultdict(list)
    for ann in annotations:
        chr_annotations[ann['seqid']].append(ann)

    # Calcular estadísticas por cromosoma
    all_te_lengths = []

    for chr_id, chr_len in chr_lengths.items():
        chr_tes = chr_annotations.get(chr_id, [])

        # Calcular bases ocupadas por TEs (sin overlaps)
        # Ordenar por posición de inicio
        sorted_tes = sorted(chr_tes, key=lambda x: x['start'])

        te_bases = 0
        last_end = 0

        for te in sorted_tes:
            # Si hay overlap con el anterior, ajustar
            if te['start'] <= last_end:
                # Overlap: solo contar la parte no solapada
                if te['end'] > last_end:
                    te_bases += te['end'] - last_end
                    last_end = te['end']
            else:
                # No overlap: contar todo
                te_bases += te['length']
                last_end = te['end']

            all_te_lengths.append(te['length'])

        background_bases = chr_len - te_bases

        stats['chromosomes'][chr_id] = {
            'chr_length': chr_len,
            'te_bases': te_bases,
            'background_bases': background_bases,
            'te_coverage_percent': (te_bases / chr_len * 100) if chr_len > 0 else 0,
            'background_percent': (background_bases / chr_len * 100) if chr_len > 0 else 0,
            'num_insertions': len(chr_tes)
        }

        stats['total_te_bases'] += te_bases

    stats['total_background_bases'] = stats['total_genome_length'] - stats['total_te_bases']

    if stats['total_genome_length'] > 0:
        stats['te_coverage_percent'] = stats['total_te_bases'] / stats['total_genome_length'] * 100
        stats['background_percent'] = stats['total_background_bases'] / stats['total_genome_length'] * 100

    if all_te_lengths:
        stats['te_length_mean'] = sum(all_te_lengths) / len(all_te_lengths)
        stats['te_length_min'] = min(all_te_lengths)
        stats['te_length_max'] = max(all_te_lengths)

    return stats


def analyze_species_genome(genome_dir: str, species_name: str, verbose: bool = True) -> Optional[Dict]:
    """
    Analizar un genoma sintético de una especie.

    Args:
        genome_dir: Directorio con los archivos del genoma
        species_name: Nombre de la especie
        verbose: Mostrar mensajes informativos

    Returns:
        Diccionario con estadísticas o None si hay error
    """
    genome_dir_path = Path(genome_dir)

    if not genome_dir_path.exists():
        if verbose:
            print(f"  ⚠️  Directorio no encontrado: {genome_dir}")
        return None

    # Buscar subdirectorio TEgenomeSimulator_*_synth_result
    synth_result_dirs = list(genome_dir_path.glob("TEgenomeSimulator_*_synth_result"))

    if not synth_result_dirs:
        # Intentar buscar directamente en el directorio actual (compatibilidad)
        search_dir = genome_dir_path
        if verbose:
            print(f"  ℹ️  No se encontró subdirectorio TEgenomeSimulator_*_synth_result")
            print(f"  ℹ️  Buscando archivos directamente en {genome_dir}")
    else:
        search_dir = synth_result_dirs[0]
        if verbose:
            print(f"  📁 Subdirectorio: {search_dir.name}")

    # Buscar archivos (TEgenomeSimulator usa patrones específicos)
    fasta_files = list(search_dir.glob("*_genome_sequence_out_final.fasta"))
    gff_files = list(search_dir.glob("*_repeat_annotation_out_final.gff"))

    if not fasta_files:
        if verbose:
            print(f"  ⚠️  No se encontró archivo FASTA en {search_dir}")
        return None

    if not gff_files:
        if verbose:
            print(f"  ⚠️  No se encontró archivo GFF en {search_dir}")
        return None

    fasta_file = str(fasta_files[0])
    gff_file = str(gff_files[0])

    if verbose:
        print(f"  📖 FASTA: {Path(fasta_file).name}")
        print(f"  📖 GFF: {Path(gff_file).name}")

    # Obtener longitudes de cromosomas
    chr_lengths = get_genome_length(fasta_file)

    if verbose:
        total_len = sum(chr_lengths.values())
        print(f"  📏 Longitud total genoma: {total_len:,} bp")
        print(f"  📏 Número de cromosomas: {len(chr_lengths)}")

    # Parsear anotaciones de TEs
    annotations = parse_gff_file(gff_file)

    if verbose:
        print(f"  🧬 Anotaciones de TEs: {len(annotations)}")

    # Calcular estadísticas
    stats = calculate_te_coverage(annotations, chr_lengths)
    stats['species'] = species_name
    stats['genome_dir'] = genome_dir

    return stats


def format_bytes(bytes_value: int) -> str:
    """Formatear bytes a unidades legibles."""
    for unit in ['bp', 'Kb', 'Mb', 'Gb']:
        if bytes_value < 1000:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1000
    return f"{bytes_value:.2f} Tb"


def print_stats_summary(stats: Dict, verbose: bool = True):
    """Imprimir resumen de estadísticas."""
    if not verbose:
        return

    print("\n" + "=" * 70)
    print(f"📊 ESTADÍSTICAS DE COVERAGE: {stats['species']}")
    print("=" * 70)

    print(f"\n🔢 TOTALES:")
    print(f"  Longitud genoma: {stats['total_genome_length']:,} bp ({format_bytes(stats['total_genome_length'])})")
    print(f"  Bases TEs: {stats['total_te_bases']:,} bp ({format_bytes(stats['total_te_bases'])})")
    print(f"  Bases background: {stats['total_background_bases']:,} bp ({format_bytes(stats['total_background_bases'])})")
    print(f"  Número de inserciones: {stats['num_te_insertions']:,}")

    print(f"\n📈 PROPORCIONES:")
    print(f"  Coverage TEs: {stats['te_coverage_percent']:.2f}%")
    print(f"  Background: {stats['background_percent']:.2f}%")
    print(f"  Ratio TE/Background: 1:{stats['background_percent']/stats['te_coverage_percent']:.2f}"
          if stats['te_coverage_percent'] > 0 else "  Ratio TE/Background: N/A")

    if stats['num_te_insertions'] > 0:
        print(f"\n📏 LONGITUD DE TEs:")
        print(f"  Media: {stats['te_length_mean']:.1f} bp")
        print(f"  Mínima: {stats['te_length_min']:,} bp")
        print(f"  Máxima: {stats['te_length_max']:,} bp")

    if len(stats['chromosomes']) > 1:
        print(f"\n🧬 POR CROMOSOMA:")
        for chr_id, chr_stats in sorted(stats['chromosomes'].items()):
            print(f"  {chr_id}:")
            print(f"    Longitud: {chr_stats['chr_length']:,} bp")
            print(f"    TEs: {chr_stats['te_coverage_percent']:.2f}% ({chr_stats['num_insertions']} inserciones)")
            print(f"    Background: {chr_stats['background_percent']:.2f}%")


def analyze_batch(input_dir: str, output_csv: str, species_filter: Optional[List[str]] = None, verbose: bool = True):
    """
    Analizar múltiples genomas sintéticos en batch.

    Args:
        input_dir: Directorio raíz con subdirectorios por especie
        output_csv: Archivo CSV de salida
        species_filter: Lista de especies a procesar (None = todas)
        verbose: Mostrar mensajes informativos
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"❌ ERROR: Directorio no encontrado: {input_dir}")
        sys.exit(1)

    # Buscar subdirectorios de especies
    species_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    if species_filter:
        species_dirs = [d for d in species_dirs if d.name in species_filter]

    if not species_dirs:
        print(f"❌ ERROR: No se encontraron directorios de especies en {input_dir}")
        sys.exit(1)

    total = len(species_dirs)

    if verbose:
        print("=" * 70)
        print(f"ANÁLISIS DE COVERAGE TE/BACKGROUND - {total} ESPECIES")
        print("=" * 70)
        print(f"Directorio entrada: {input_dir}")
        print(f"Archivo salida: {output_csv}")
        print("=" * 70)

    # Procesar cada especie
    all_stats = []
    success_count = 0
    failed_species = []

    for idx, species_dir in enumerate(sorted(species_dirs), 1):
        species_name = species_dir.name

        if verbose:
            print(f"\n[{idx}/{total}] 🔍 Analizando: {species_name}")

        stats = analyze_species_genome(str(species_dir), species_name, verbose=verbose)

        if stats:
            all_stats.append(stats)
            success_count += 1

            if verbose:
                print_stats_summary(stats, verbose=False)  # Resumen breve
                print(f"  ✅ Coverage TEs: {stats['te_coverage_percent']:.2f}%")
        else:
            failed_species.append(species_name)

    # Guardar resultados en CSV
    if all_stats:
        fieldnames = [
            'species',
            'total_genome_length',
            'total_te_bases',
            'total_background_bases',
            'num_te_insertions',
            'te_coverage_percent',
            'background_percent',
            'te_length_mean',
            'te_length_min',
            'te_length_max',
            'genome_dir'
        ]

        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_stats)

        if verbose:
            print("\n" + "=" * 70)
            print("📊 REPORTE FINAL")
            print("=" * 70)
            print(f"  Total procesadas: {total}")
            print(f"  ✅ Exitosas: {success_count} ({100*success_count/total:.1f}%)")
            print(f"  ❌ Fallidas: {len(failed_species)} ({100*len(failed_species)/total:.1f}%)")

            if failed_species:
                print(f"\n  ⚠️  Especies sin analizar:")
                for species in failed_species[:10]:
                    print(f"    - {species}")
                if len(failed_species) > 10:
                    print(f"    ... y {len(failed_species)-10} más")

            # Estadísticas globales
            if all_stats:
                avg_te_coverage = sum(s['te_coverage_percent'] for s in all_stats) / len(all_stats)
                min_te_coverage = min(s['te_coverage_percent'] for s in all_stats)
                max_te_coverage = max(s['te_coverage_percent'] for s in all_stats)

                print(f"\n  📈 Estadísticas de Coverage TE:")
                print(f"    Media: {avg_te_coverage:.2f}%")
                print(f"    Mínimo: {min_te_coverage:.2f}%")
                print(f"    Máximo: {max_te_coverage:.2f}%")

                # Top 5 especies con más coverage de TEs
                print(f"\n  🏆 TOP 5 especies con mayor coverage de TEs:")
                sorted_stats = sorted(all_stats, key=lambda x: x['te_coverage_percent'], reverse=True)
                for i, s in enumerate(sorted_stats[:5], 1):
                    print(f"    {i}. {s['species']:35} {s['te_coverage_percent']:6.2f}%")

                # Bottom 5 especies con menos coverage de TEs
                print(f"\n  📉 TOP 5 especies con menor coverage de TEs:")
                for i, s in enumerate(sorted_stats[-5:][::-1], 1):
                    print(f"    {i}. {s['species']:35} {s['te_coverage_percent']:6.2f}%")

            print("\n" + "=" * 70)
            print(f"💾 Resultados guardados en: {output_csv}")
            print("=" * 70)

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description='Analizar proporción TE/Background en genomas sintéticos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPCIÓN:
  Este script analiza los genomas sintéticos generados con TEgenomeSimulator
  y calcula la proporción real entre secuencia background y TEs insertados.

CARACTERÍSTICAS:
  - Lee archivos FASTA para obtener longitud total del genoma
  - Parsea archivos GFF con anotaciones de TEs
  - Calcula coverage de TEs (con corrección de overlaps)
  - Genera estadísticas detalladas por especie
  - Exporta resultados a CSV

EJEMPLOS DE USO:
  # Analizar una sola especie
  python analyze_te_coverage.py \\
      --species-dir synthetic_genomes/Oryza_sativa/ \\
      --species-name "Oryza_sativa"

  # Analizar todas las especies en batch
  python analyze_te_coverage.py \\
      --batch-dir synthetic_genomes/ \\
      --output te_coverage_stats.csv

  # Analizar especies específicas
  python analyze_te_coverage.py \\
      --batch-dir synthetic_genomes/ \\
      --output te_coverage_stats.csv \\
      --species "Oryza_sativa,Arabidopsis_thaliana,Drosophila_melanogaster"

OUTPUTS:
  - CSV con estadísticas por especie
  - Resumen en consola con:
    * Longitud total del genoma
    * Bases ocupadas por TEs
    * Bases de background
    * Porcentajes de coverage
    * Estadísticas de longitud de TEs
        """
    )

    parser.add_argument('--species-dir',
                       help='Directorio de una especie específica')
    parser.add_argument('--species-name',
                       help='Nombre de la especie (requerido con --species-dir)')
    parser.add_argument('--batch-dir',
                       help='Directorio raíz con subdirectorios de especies')
    parser.add_argument('--output', '-o', default='te_coverage_stats.csv',
                       help='Archivo CSV de salida (default: te_coverage_stats.csv)')
    parser.add_argument('--species',
                       help='Lista de especies separadas por comas (solo con --batch-dir)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Modo silencioso')

    args = parser.parse_args()

    verbose = not args.quiet

    # Validar argumentos
    if not args.species_dir and not args.batch_dir:
        parser.error("Debe especificar --species-dir o --batch-dir")

    if args.species_dir and args.batch_dir:
        parser.error("No puede usar --species-dir y --batch-dir simultáneamente")

    if args.species_dir and not args.species_name:
        parser.error("--species-name es requerido con --species-dir")

    # Modo single species
    if args.species_dir:
        stats = analyze_species_genome(args.species_dir, args.species_name, verbose=verbose)

        if stats:
            print_stats_summary(stats, verbose=True)

            # Guardar en CSV también
            with open(args.output, 'w', newline='') as f:
                fieldnames = list(stats.keys())
                # Excluir campos complejos
                fieldnames = [f for f in fieldnames if f != 'chromosomes']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({k: v for k, v in stats.items() if k != 'chromosomes'})

            if verbose:
                print(f"\n💾 Estadísticas guardadas en: {args.output}")
        else:
            print("❌ ERROR: No se pudo analizar el genoma")
            sys.exit(1)

    # Modo batch
    else:
        species_filter = None
        if args.species:
            species_filter = [s.strip() for s in args.species.split(',')]

        analyze_batch(args.batch_dir, args.output, species_filter=species_filter, verbose=verbose)


if __name__ == "__main__":
    main()
