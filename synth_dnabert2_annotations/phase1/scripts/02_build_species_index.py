#!/usr/bin/env python3
"""
Generar índice de especies sin duplicar archivos
Autor: Jorge González Gilbaja
TFM - Detección automática de TEs mediante Deep Learning
Fecha: 16/10/2025

Este script genera un índice de especies que permite filtrar secuencias
en tiempo de ejecución sin necesidad de crear archivos individuales.

ESTRATEGIA:
1. Mantener InpactorDB2 como archivo único
2. Generar índice: especie -> IDs de secuencias
3. Filtrar on-the-fly durante generación de genomas
"""

import argparse
import re
import pandas as pd
from collections import defaultdict
from Bio import SeqIO
import json

def parse_header(header):
    """Parsear header para extraer especie."""
    info = {
        'full_header': header,
        'species': None,
        'id': None,
        'family': None
    }
    
    # ID
    parts = header.split(None, 1)
    if parts:
        info['id'] = parts[0].lstrip('>')
    
    # Especie (último elemento)
    species_match = re.search(r'\s([A-Z][a-z]+\s+[a-z]+)$', header)
    if species_match:
        info['species'] = species_match.group(1)
    
    # Familia
    class_match = re.search(r'#([^\s]+)', header)
    if class_match:
        classification = class_match.group(1)
        family_parts = classification.split('/')
        if len(family_parts) >= 3:
            info['family'] = family_parts[2]
    
    return info

def build_species_index(input_fasta, species_gc_csv=None, min_families=None, verbose=True):
    """
    Construir índice de especies sin duplicar archivos.

    Returns:
        dict: {
            'species_name': {
                'sequences': [{'id': str, 'species': str}, ...],
                'families': list de familias,
                'num_sequences': int,
                'num_families': int,
                'gc_percent': float (si disponible)
            }
        }
    """
    
    if verbose:
        print("=" * 70)
        print("GENERANDO ÍNDICE DE ESPECIES")
        print("=" * 70)
        print(f"Archivo FASTA: {input_fasta}")
        if species_gc_csv:
            print(f"Datos de GC: {species_gc_csv}")
        print("=" * 70)
    
    # Cargar datos de GC si están disponibles
    gc_data = {}
    valid_species = None
    
    if species_gc_csv:
        df_gc = pd.read_csv(species_gc_csv)
        
        # Filtrar por familias si se especifica
        if min_families and 'num_families' in df_gc.columns:
            df_gc = df_gc[df_gc['num_families'] >= min_families]
            if verbose:
                print(f"Filtrando especies con >= {min_families} familias")
        
        # Crear diccionario de GC
        if 'gc_percent' in df_gc.columns:
            for _, row in df_gc.iterrows():
                # Priorizar gc_percent, si no está disponible usar gc_median
                if pd.notna(row.get('gc_percent')):
                    gc_data[row['species']] = float(row['gc_percent'])
                elif pd.notna(row.get('gc_median')):
                    gc_data[row['species']] = float(row['gc_median'])
        
        valid_species = set(df_gc['species'].tolist())
        if verbose:
            print(f"Especies válidas con GC: {len(valid_species)}")
    
    # Construir índice
    species_index = defaultdict(lambda: {
        'sequences': [],
        'families': set(),
        'num_sequences': 0
    })
    
    total_sequences = 0
    skipped = 0
    
    if verbose:
        print("\n📖 Leyendo secuencias y construyendo índice...")
    
    for record in SeqIO.parse(input_fasta, "fasta"):
        total_sequences += 1
        
        info = parse_header(record.description)
        species = info['species']

        if not species:
            skipped += 1
            continue
        
        # Filtrar por especies válidas si se especificó
        if valid_species and species not in valid_species:
            skipped += 1
            continue
        
        # Agregar al índice (almacenar diccionario con id y species)
        species_index[species]['sequences'].append({
            'id': info['id'],
            'species': species
        })
        species_index[species]['num_sequences'] += 1
        
        if info['family']:
            species_index[species]['families'].add(info['family'])
        
        if verbose and total_sequences % 10000 == 0:
            print(f"  Procesadas {total_sequences:,} secuencias, "
                  f"{len(species_index)} especies...")
    
    # Añadir datos de GC
    for species in species_index:
        if species in gc_data:
            species_index[species]['gc_percent'] = gc_data[species]
        else:
            species_index[species]['gc_percent'] = None
        
        # Convertir set a lista para JSON
        species_index[species]['families'] = sorted(list(species_index[species]['families']))
        species_index[species]['num_families'] = len(species_index[species]['families'])
    
    if verbose:
        print("=" * 70)
        print(f"✅ Índice completado:")
        print(f"   Total secuencias procesadas: {total_sequences:,}")
        print(f"   Secuencias indexadas: {total_sequences - skipped:,}")
        print(f"   Secuencias omitidas: {skipped:,}")
        print(f"   Especies en índice: {len(species_index)}")
        print("=" * 70)
    
    return dict(species_index)

def save_index(species_index, output_file, verbose=True):
    """Guardar índice en formato JSON."""
    
    with open(output_file, 'w') as f:
        json.dump(species_index, f, indent=2)
    
    if verbose:
        print(f"\n💾 Índice guardado: {output_file}")
        
        # Estadísticas del archivo
        import os
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"   Tamaño: {size_mb:.2f} MB")

def save_summary_csv(species_index, output_file, verbose=True):
    """Guardar resumen en CSV para fácil visualización."""
    
    records = []
    for species, data in species_index.items():
        records.append({
            'species': species,
            'num_sequences': data['num_sequences'],
            'num_families': data['num_families'],
            'gc_percent': data.get('gc_percent'),
            'families': ','.join(data['families'])
        })
    
    df = pd.DataFrame(records)
    df = df.sort_values('num_sequences', ascending=False)
    df.to_csv(output_file, index=False)
    
    if verbose:
        print(f"📊 Resumen CSV guardado: {output_file}")
        print(f"\n🏆 TOP 10 ESPECIES:")
        for idx, row in df.head(10).iterrows():
            gc_str = f"{row['gc_percent']:.2f}%" if pd.notna(row['gc_percent']) else "N/A"
            print(f"   {row['species']:35} {row['num_sequences']:6,} seqs | "
                  f"{row['num_families']:3} fam | GC: {gc_str}")

def main():
    parser = argparse.ArgumentParser(
        description='Generar índice de especies sin duplicar archivos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ESTRATEGIA MEJORADA:
  En lugar de generar ~1,500 archivos FASTA individuales, este script:
  1. Genera un índice JSON con los IDs de secuencias por especie
  2. Mantiene el archivo original sin duplicar
  3. Permite filtrado on-the-fly durante generación de genomas

VENTAJAS:
  - Solo 2-3 archivos en total (vs ~1,500)
  - Sin duplicación de datos
  - Más rápido y eficiente
  - Fácil de gestionar y respaldar

EJEMPLOS DE USO:
  # Generar índice completo
  python build_species_index.py \\
      --input r.1.5_all.fasta \\
      --species-gc species_gc_data.csv \\
      --output species_index.json
  
  # Con filtrado por familias
  python build_species_index.py \\
      --input r.1.5_all.fasta \\
      --species-gc species_gc_data.csv \\
      --min-families 5 \\
      --output species_index.json

OUTPUTS:
  - species_index.json: Índice completo (para uso programático)
  - species_index_summary.csv: Resumen legible
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Archivo FASTA de InpactorDB2')
    parser.add_argument('--output', '-o', required=True,
                       help='Archivo JSON de salida con índice')
    parser.add_argument('--species-gc', '-g',
                       help='CSV con datos de especies y GC')
    parser.add_argument('--min-families', type=int,
                       help='Mínimo número de familias para incluir especie')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Modo silencioso')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Construir índice
    species_index = build_species_index(
        args.input,
        species_gc_csv=args.species_gc,
        min_families=args.min_families,
        verbose=verbose
    )
    
    # Guardar índice JSON
    save_index(species_index, args.output, verbose=verbose)
    
    # Guardar resumen CSV
    summary_file = args.output.replace('.json', '_summary.csv')
    save_summary_csv(species_index, summary_file, verbose=verbose)
    
    if verbose:
        print("\n✅ Proceso completado!")
        print("\n📋 Próximos pasos:")
        print("  1. Usar species_index.json en TEgenomeSimulator (filtrado on-the-fly)")
        print("  2. Revisar species_index_summary.csv para estadísticas")
        print("  3. Generar chr_index.csv directamente desde summary")

if __name__ == "__main__":
    main()
