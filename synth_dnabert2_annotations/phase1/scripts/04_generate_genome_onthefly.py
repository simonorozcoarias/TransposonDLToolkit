#!/usr/bin/env python3
"""
Wrapper para TEgenomeSimulator con filtrado on-the-fly
Autor: Jorge González Gilbaja
TFM - Detección automática de TEs mediante Deep Learning
Fecha: 16/10/2025

Este script filtra secuencias de una especie específica del archivo
InpactorDB2 completo y ejecuta TEgenomeSimulator sin crear archivos intermedios.
"""

import argparse
import json
import csv
import sys
import os
import tempfile
import subprocess
import re
from Bio import SeqIO

MAX_COPIES_LIMIT = 500

def parse_header(header):
    """
    Parsear header de FASTA para extraer especie.
    Misma función que en build_species_index.py para consistencia.
    """
    info = {
        'species': None,
        'id': None
    }

    # ID
    parts = header.split(None, 1)
    if parts:
        info['id'] = parts[0].lstrip('>')

    # Especie (último elemento)
    species_match = re.search(r'\s([A-Z][a-z]+\s+[a-z]+)$', header)
    if species_match:
        info['species'] = species_match.group(1)

    return info

def filter_sequences_for_species(input_fasta, species_index_file, species_name, output_fasta):
    """
    Filtrar secuencias de una especie específica usando el índice.
    
    Args:
        input_fasta: Archivo FASTA completo de InpactorDB2
        species_index_file: Archivo JSON con índice de especies
        species_name: Nombre de la especie a filtrar
        output_fasta: Archivo temporal de salida
    
    Returns:
        int: Número de secuencias filtradas
    """
    
    # Cargar índice
    with open(species_index_file, 'r') as f:
        species_index = json.load(f)
    
    if species_name not in species_index:
        print(f"❌ ERROR: Especie no encontrada en índice: {species_name}")
        print(f"   Especies disponibles: {len(species_index)}")
        return 0
    
    # Construir set de (id, species) para validación correcta
    # IMPORTANTE: Esto previene que se incluyan secuencias de otras especies
    # que puedan compartir el mismo ID (ej: MER34A1_2 en Cavia y Eutheria)
    sequence_lookup = set()
    for seq_data in species_index[species_name]['sequences']:
        sequence_lookup.add((seq_data['id'], seq_data['species']))

    print(f"🔍 Filtrando {len(sequence_lookup)} secuencias de {species_name}...")

    # Filtrar secuencias validando TANTO id COMO species
    filtered_count = 0
    filtered_records = []
    skipped_count = 0

    for record in SeqIO.parse(input_fasta, "fasta"):
        # Extraer especie del header completo
        info = parse_header(record.description)

        # Validar que coincidan TANTO el ID COMO la especie
        if (record.id, info['species']) in sequence_lookup:
            filtered_records.append(record)
            filtered_count += 1

            if filtered_count % 1000 == 0:
                print(f"  Filtradas: {filtered_count}/{len(sequence_lookup)}")
        # Detectar IDs duplicados de otras especies (para logging)
        elif record.id in [seq_data['id'] for seq_data in species_index[species_name]['sequences']]:
            skipped_count += 1
    
    # Escribir archivo temporal
    SeqIO.write(filtered_records, output_fasta, "fasta")

    print(f"✅ Filtrado completado: {filtered_count} secuencias")
    if skipped_count > 0:
        print(f"   ℹ️  Omitidas {skipped_count} secuencias con ID duplicado de otras especies")
    return filtered_count

def get_species_data_from_csv(csv_file, species_name):
    """
    Leer datos de una especie del CSV de species_gc_data.

    Args:
        csv_file: Ruta al CSV de species_gc_data
        species_name: Nombre de la especie a buscar

    Returns:
        dict: Datos de la especie o None si no se encuentra
    """
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['species'] == species_name:
                return row
    return None

def create_chr_index(species_name, gc_percent, output_file, chr_size=50000000):
    """Crear archivo chr_index.csv temporal."""

    chr_name = f"chr_synthetic_{species_name.replace(' ', '_')}_01"

    with open(output_file, 'w') as f:
        f.write(f"{chr_name},{chr_size},{gc_percent:.2f}\n")

def run_tegenome_simulator(species_name, te_library, chr_index, output_dir,
                          max_copies, min_copies=5,
                          min_identity=85, max_identity=95,
                          min_sd=5, max_sd=15, simulator_dir='.'):
    """Ejecutar TEgenomeSimulator."""

    te_simulator_path = os.path.join(simulator_dir, 'TEgenomeSimulator/TEgenomeSimulator.py')

    cmd = [
        'python3', te_simulator_path,
        '-M', '0',
        '-p', f'{species_name.replace(" ", "_")}_synth',
        '-c', chr_index,
        '-r', te_library,
        '-m', str(max_copies),
        '-n', str(min_copies),
        # '--minidn', str(min_identity),
        # '--maxidn', str(max_identity),
        # '--minsd', str(min_sd),
        # '--maxsd', str(max_sd),
        '-o', output_dir,
        '-t', '8'
    ]
    
    print(f"\n🚀 Ejecutando TEgenomeSimulator...")
    print(f"   Comando: {' '.join(cmd)}")
    sys.stdout.flush()  # Asegurar que el print se escribe antes del subprocess

    # No capturar output para que fluya directamente a los archivos de SLURM
    result = subprocess.run(cmd, text=True)

    return result

def main():
    parser = argparse.ArgumentParser(
        description='Generar genoma sintético con filtrado on-the-fly',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ESTRATEGIA:
  Este script evita crear ~1,500 archivos FASTA individuales:
  1. Lee el índice de especies (JSON) para filtrado
  2. Lee datos de GC y copies del CSV (de NCBI)
  3. Filtra secuencias de la especie on-the-fly
  4. Crea archivo temporal solo durante la ejecución
  5. Ejecuta TEgenomeSimulator
  6. Limpia archivos temporales

VENTAJAS:
  - Sin duplicación de datos
  - Solo 2-3 archivos permanentes
  - Filtrado rápido gracias al índice
  - Usa %GC de genomas de referencia (NCBI)
  - Cálculo automático de max/min copies

EJEMPLO DE USO:
  # Con valores calculados automáticamente
  python generate_genome_onthefly.py \\
      --species "Oryza_sativa" \\
      --input-fasta r.1.5_all.fasta \\
      --species-index species_index.json \\
      --species-gc-data species_gc_data.csv \\
      --output-dir synthetic_genomes/Oryza_sativa/

  # Con override manual de max-copies
  python generate_genome_onthefly.py \\
      --species "Oryza_sativa" \\
      --input-fasta r.1.5_all.fasta \\
      --species-index species_index.json \\
      --species-gc-data species_gc_data.csv \\
      --output-dir synthetic_genomes/Oryza_sativa/ \\
      --max-copies 1500
        """
    )
    
    parser.add_argument('--species', required=True,
                       help='Nombre de la especie a procesar')
    parser.add_argument('--input-fasta', required=True,
                       help='Archivo FASTA completo de InpactorDB2')
    parser.add_argument('--species-index', required=True,
                       help='Archivo JSON con índice de especies')
    parser.add_argument('--species-gc-data', required=True,
                       help='CSV con datos de GC y max/min copies (output de get_species_gc_optimized)')
    parser.add_argument('--output-dir', required=True,
                       help='Directorio de salida')
    parser.add_argument('--max-copies', type=int,
                       help='Número máximo de copias por familia (-m). Si no se especifica, se usa el valor calculado del CSV')
    parser.add_argument('--min-copies', type=int,
                       help='Número mínimo de copias (-n). Si no se especifica, se usa el valor calculado del CSV o default 5')
    parser.add_argument('--min-identity', type=int, default=85,
                       help='Identidad mínima (default: 85)')
    parser.add_argument('--max-identity', type=int, default=95,
                       help='Identidad máxima (default: 95)')
    parser.add_argument('--keep-temp', action='store_true',
                       help='No eliminar archivos temporales')
    parser.add_argument('--simulator-dir', default='.',
                       help='Directorio donde se encuentra el repositorio TEgenomeSimulator (default: directorio actual)')
    parser.add_argument('--scaling-factor', type=float, default=1.0,
                       help='Factor de escalado para ajustar max/min copies del CSV (default: 1.0). '
                            'Útil para compensar la fragmentación de TEgenomeSimulator. '
                            'Ejemplo: --scaling-factor 1.5 aumenta los copies en 50%%')

    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)

    # Cargar datos de la especie del CSV de species_gc_data
    species_data = get_species_data_from_csv(args.species_gc_data, args.species)

    if not species_data:
        print(f"❌ ERROR: Especie no encontrada en {args.species_gc_data}: {args.species}")
        sys.exit(1)

    # Obtener gc_percent del CSV (de NCBI, prioritario)
    gc_percent_str = species_data.get('gc_percent', '').strip()
    if not gc_percent_str or gc_percent_str == 'N/A' or gc_percent_str == 'None':
        print(f"❌ ERROR: No hay %GC disponible para {args.species}")
        print(f"   Esta especie no tiene datos de genoma de referencia en NCBI")
        print(f"   Especie descartada")
        sys.exit(1)

    try:
        gc_percent = float(gc_percent_str)
    except ValueError:
        print(f"❌ ERROR: %GC inválido para {args.species}: {gc_percent_str}")
        sys.exit(1)

    # Cargar species_index para obtener sequence_ids (necesario para filtrado)
    with open(args.species_index, 'r') as f:
        species_index = json.load(f)

    if args.species not in species_index:
        print(f"❌ ERROR: Especie no encontrada en species_index: {args.species}")
        sys.exit(1)

    # Determinar max_copies y min_copies
    # Prioridad: CLI > CSV (con scaling factor aplicado)
    if args.max_copies is not None:
        max_copies = args.max_copies
        max_copies_source = "CLI (override)"
    else:
        max_copies_str = species_data.get('max_copies_calculated', '').strip()
        if max_copies_str and max_copies_str != 'None':
            try:
                max_copies_base = int(float(max_copies_str))
                # Aplicar scaling factor
                max_copies = int(max_copies_base * args.scaling_factor)
                if max_copies > MAX_COPIES_LIMIT:
                    max_copies = MAX_COPIES_LIMIT
                    max_copies_source = "MAX_COPIES_LIMIT"
                else:
                    if args.scaling_factor != 1.0:
                        max_copies_source = f"CSV × {args.scaling_factor}"
                    else:
                        max_copies_source = "CSV (calculado)"
            except ValueError:
                print(f"❌ ERROR: max_copies_calculated inválido: {max_copies_str}")
                sys.exit(1)
        else:
            print(f"❌ ERROR: No hay max_copies calculado y no se especificó por CLI")
            sys.exit(1)

    if args.min_copies is not None and args.min_copies != 5:  # 5 es el default
        min_copies = args.min_copies
        min_copies_source = "CLI (override)"
    else:
        min_copies_str = species_data.get('min_copies_calculated', '').strip()
        if min_copies_str and min_copies_str != 'None':
            try:
                min_copies_base = int(float(min_copies_str))
                # Aplicar scaling factor
                min_copies = int(min_copies_base * args.scaling_factor)
                if min_copies > MAX_COPIES_LIMIT:
                    min_copies = MAX_COPIES_LIMIT
                    min_copies_source = "MAX_COPIES_LIMIT"
                else:
                    if args.scaling_factor != 1.0:
                        min_copies_source = f"CSV × {args.scaling_factor}"
                    else:
                        min_copies_source = "CSV (calculado)"
            except ValueError:
                min_copies = 5
                min_copies_source = "default (5)"
        else:
            min_copies = 5
            min_copies_source = "default (5)"

    print("=" * 70)
    print(f"GENERANDO GENOMA SINTÉTICO: {args.species}")
    print("=" * 70)
    print(f"Secuencias: {species_index[args.species]['num_sequences']}")
    print(f"Familias: {species_index[args.species]['num_families']}")
    print(f"GC: {gc_percent:.2f}% (NCBI)")
    if args.scaling_factor != 1.0:
        print(f"Scaling factor: {args.scaling_factor}")
    print(f"Max copias: {max_copies} ({max_copies_source})")
    print(f"Min copias: {min_copies} ({min_copies_source})")
    print("=" * 70)
    
    # Crear archivos temporales
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as temp_fasta:
        temp_fasta_path = temp_fasta.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_chr:
        temp_chr_path = temp_chr.name
    
    try:
        # 1. Filtrar secuencias
        num_seqs = filter_sequences_for_species(
            args.input_fasta,
            args.species_index,
            args.species,
            temp_fasta_path
        )
        
        if num_seqs == 0:
            print("❌ No se filtraron secuencias")
            sys.exit(1)
        
        # 2. Crear chr_index
        create_chr_index(args.species, gc_percent, temp_chr_path)
        print(f"✅ chr_index creado: {temp_chr_path}")
        
        # 3. Ejecutar TEgenomeSimulator
        result = run_tegenome_simulator(
            args.species,
            temp_fasta_path,
            temp_chr_path,
            args.output_dir,
            max_copies,
            min_copies=min_copies,
            min_identity=args.min_identity,
            max_identity=args.max_identity,
            simulator_dir=args.simulator_dir
        )
        
        if result.returncode == 0:
            print("\n✅ Genoma generado exitosamente!")
            print(f"   Output: {args.output_dir}")
        else:
            print("\n❌ Error ejecutando TEgenomeSimulator")
            print(f"   Return code: {result.returncode}")
            print(f"   Ver detalles en los archivos de log de SLURM")
            sys.exit(1)
    
    finally:
        # Limpiar archivos temporales
        if not args.keep_temp:
            if os.path.exists(temp_fasta_path):
                os.unlink(temp_fasta_path)
                print(f"🗑️  Eliminado: {temp_fasta_path}")
            if os.path.exists(temp_chr_path):
                os.unlink(temp_chr_path)
                print(f"🗑️  Eliminado: {temp_chr_path}")
        else:
            print(f"\n📁 Archivos temporales preservados:")
            print(f"   TE library: {temp_fasta_path}")
            print(f"   chr_index: {temp_chr_path}")

if __name__ == "__main__":
    main()
