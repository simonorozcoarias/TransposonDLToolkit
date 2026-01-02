#!/usr/bin/env python3
"""
Extraer especies únicas de InpactorDB2 y generar estadísticas
Autor: Jorge González Gilbaja
TFM - Detección automática de TEs mediante Deep Learning
Fecha: 14/10/2025

Este script analiza el archivo completo de InpactorDB2 para:
1. Identificar todas las especies únicas
2. Contar familias de TEs por especie
3. Calcular estadísticas (longitudes, GC, número de secuencias)
4. Generar CSV con toda la información
"""

import argparse
import re
from collections import defaultdict
from Bio import SeqIO
import pandas as pd

def parse_header(header):
    """
    Parsear el header del FASTA para extraer información relevante.
    Formato esperado: >rnd-X_family-Y_Z#CLASSI/SUPERFAMILY/FAMILY Species
    """
    info = {
        'full_header': header,
        'family': None,
        'superfamily': None,
        'order': None,
        'species': None
    }
    
    # Extraer clasificación (después de #)
    class_match = re.search(r'#([^\s]+)', header)
    if class_match:
        classification = class_match.group(1)
        parts = classification.split('/')
        
        if len(parts) >= 1:
            info['order'] = parts[0]  # CLASSI o CLASSII
        if len(parts) >= 2:
            info['superfamily'] = parts[1]  # LTR, LINE, etc.
        if len(parts) >= 3:
            info['family'] = parts[2]  # GYPSY, COPIA, etc.
    
    # Extraer especie (las dos últimas palabras separadas por espacio)
    species_match = re.search(r'\s([A-Z][a-z]+\s+[a-z]+)$', header)
    if species_match:
        info['species'] = species_match.group(1)
    
    return info

def analyze_inpactordb2(fasta_file, verbose=True):
    """
    Analizar InpactorDB2 y extraer estadísticas por especie.
    """
    # Estructura de datos por especie
    species_data = defaultdict(lambda: {
        'families': set(),
        'superfamilies': set(),
        'orders': set(),
        'num_sequences': 0,
        'lengths': [],
        'gc_contents': []
    })
    
    total_sequences = 0
    
    if verbose:
        print(f"📖 Analizando {fasta_file}...")
        print("=" * 70)
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        total_sequences += 1
        
        # Parsear header
        info = parse_header(record.description)
        species = info['species']
        
        if not species:
            if verbose and total_sequences % 10000 == 0:
                print(f"⚠️  Secuencia sin especie: {record.id}")
            continue
        
        # Calcular estadísticas de la secuencia
        seq = str(record.seq)
        length = len(seq)
        gc = (seq.upper().count('G') + seq.upper().count('C')) / length * 100
        
        # Agregar a datos de la especie
        species_data[species]['num_sequences'] += 1
        species_data[species]['lengths'].append(length)
        species_data[species]['gc_contents'].append(gc)
        
        if info['family']:
            species_data[species]['families'].add(info['family'])
        if info['superfamily']:
            species_data[species]['superfamilies'].add(info['superfamily'])
        if info['order']:
            species_data[species]['orders'].add(info['order'])
        
        if verbose and total_sequences % 10000 == 0:
            print(f"  Procesadas {total_sequences:,} secuencias, {len(species_data)} especies únicas...")
    
    if verbose:
        print("=" * 70)
        print(f"✅ Análisis completado: {total_sequences:,} secuencias procesadas")
        print(f"✅ Especies únicas encontradas: {len(species_data)}")
    
    return species_data, total_sequences

def generate_statistics_dataframe(species_data):
    """
    Generar DataFrame con estadísticas por especie.
    """
    records = []
    
    for species, data in species_data.items():
        lengths = data['lengths']
        gc_contents = data['gc_contents']
        
        record = {
            'species': species,
            'num_sequences': data['num_sequences'],
            'num_families': len(data['families']),
            'num_superfamilies': len(data['superfamilies']),
            'num_orders': len(data['orders']),
            'families': ','.join(sorted(data['families'])),
            'superfamilies': ','.join(sorted(data['superfamilies'])),
            'orders': ','.join(sorted(data['orders'])),
            'length_min': min(lengths) if lengths else 0,
            'length_max': max(lengths) if lengths else 0,
            'length_mean': sum(lengths) / len(lengths) if lengths else 0,
            'length_median': sorted(lengths)[len(lengths)//2] if lengths else 0,
            'gc_mean': sum(gc_contents) / len(gc_contents) if gc_contents else 0,
            'gc_median': sorted(gc_contents)[len(gc_contents)//2] if gc_contents else 0
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Ordenar por número de secuencias (descendente)
    df = df.sort_values('num_sequences', ascending=False)
    
    return df

def print_summary_report(df, total_sequences):
    """
    Imprimir reporte resumen del análisis.
    """
    print("\n" + "=" * 70)
    print("📊 REPORTE DE ESPECIES EN INPACTORDB2")
    print("=" * 70)
    
    print(f"\n🔢 ESTADÍSTICAS GENERALES")
    print(f"  Total de secuencias: {total_sequences:,}")
    print(f"  Total de especies: {len(df)}")
    print(f"  Secuencias por especie (promedio): {df['num_sequences'].mean():.1f}")
    print(f"  Secuencias por especie (mediana): {df['num_sequences'].median():.1f}")
    
    print(f"\n🧬 DISTRIBUCIÓN DE FAMILIAS")
    print(f"  Familias por especie (promedio): {df['num_families'].mean():.1f}")
    print(f"  Familias por especie (mediana): {df['num_families'].median():.1f}")
    print(f"  Familias por especie (máximo): {df['num_families'].max()}")
    print(f"  Familias por especie (mínimo): {df['num_families'].min()}")
    
    print(f"\n📏 ESTADÍSTICAS DE LONGITUD")
    print(f"  Longitud promedio global: {df['length_mean'].mean():.1f} bp")
    print(f"  Longitud mínima global: {df['length_min'].min()} bp")
    print(f"  Longitud máxima global: {df['length_max'].max()} bp")
    
    print(f"\n🧪 CONTENIDO GC")
    print(f"  GC promedio global: {df['gc_mean'].mean():.2f}%")
    print(f"  GC rango: {df['gc_mean'].min():.2f}% - {df['gc_mean'].max():.2f}%")
    
    print(f"\n🏆 TOP 10 ESPECIES (por número de secuencias)")
    print("-" * 70)
    top10 = df.head(10)
    for idx, row in top10.iterrows():
        print(f"  {row['species']:30} {row['num_sequences']:6,} seqs | "
              f"{row['num_families']:3} familias | "
              f"GC: {row['gc_mean']:5.2f}%")
    
    print(f"\n⚠️  ESPECIES CON POCAS FAMILIAS (<5)")
    low_families = df[df['num_families'] < 5]
    if len(low_families) > 0:
        print(f"  Total: {len(low_families)} especies")
        if len(low_families) <= 10:
            for idx, row in low_families.iterrows():
                print(f"    {row['species']:30} {row['num_families']} familias | "
                      f"{row['num_sequences']} secuencias")
        else:
            print(f"    (Ver archivo CSV para lista completa)")
    else:
        print("  ✅ Todas las especies tienen ≥5 familias")
    
    print("\n" + "=" * 70)

def filter_species(df, min_families=None, min_sequences=None):
    """
    Filtrar especies según criterios.
    """
    filtered = df.copy()
    
    if min_families:
        before = len(filtered)
        filtered = filtered[filtered['num_families'] >= min_families]
        print(f"  Filtro mínimo {min_families} familias: {before} → {len(filtered)} especies")
    
    if min_sequences:
        before = len(filtered)
        filtered = filtered[filtered['num_sequences'] >= min_sequences]
        print(f"  Filtro mínimo {min_sequences} secuencias: {before} → {len(filtered)} especies")
    
    return filtered

def main():
    parser = argparse.ArgumentParser(
        description='Extraer y analizar especies de InpactorDB2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EJEMPLOS DE USO:
  # Análisis básico
  python extract_species_from_inpactordb2.py -i r.1.5_all.fasta -o species_list.csv
  
  # Con filtrado
  python extract_species_from_inpactordb2.py -i r.1.5_all.fasta -o species_list.csv \\
      --min-families 5 --min-sequences 100
  
  # Generar lista filtrada adicional
  python extract_species_from_inpactordb2.py -i r.1.5_all.fasta -o species_list.csv \\
      --filtered-output species_filtered.csv --min-families 10
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Archivo FASTA de InpactorDB2')
    parser.add_argument('--output', '-o', required=True,
                       help='Archivo CSV de salida con estadísticas por especie')
    parser.add_argument('--filtered-output', '-f',
                       help='Archivo CSV adicional con especies filtradas')
    parser.add_argument('--min-families', type=int,
                       help='Mínimo número de familias para filtrar')
    parser.add_argument('--min-sequences', type=int,
                       help='Mínimo número de secuencias para filtrar')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Modo silencioso (menos output)')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if verbose:
        print("=" * 70)
        print("ANÁLISIS DE ESPECIES EN INPACTORDB2")
        print("=" * 70)
        print(f"Archivo de entrada: {args.input}")
        print(f"Archivo de salida: {args.output}")
        print("=" * 70)
    
    # Analizar InpactorDB2
    species_data, total_sequences = analyze_inpactordb2(args.input, verbose=verbose)
    
    # Generar DataFrame
    df = generate_statistics_dataframe(species_data)
    
    # Imprimir reporte
    if verbose:
        print_summary_report(df, total_sequences)
    
    # Guardar CSV completo
    df.to_csv(args.output, index=False)
    print(f"\n💾 Guardado: {args.output} ({len(df)} especies)")
    
    # Generar CSV filtrado si se solicita
    if args.filtered_output:
        print(f"\n🔍 Aplicando filtros...")
        df_filtered = filter_species(df, args.min_families, args.min_sequences)
        df_filtered.to_csv(args.filtered_output, index=False)
        print(f"💾 Guardado: {args.filtered_output} ({len(df_filtered)} especies)")
    
    print("\n✅ Análisis completado exitosamente!")
    print("\n📋 Próximos pasos:")
    print("  1. Revisar species_list.csv")
    print("  2. Decidir criterios de filtrado (si aplica)")
    print("  3. Proceder con T1.2: Obtención de %GC por especie")

if __name__ == "__main__":
    main()
