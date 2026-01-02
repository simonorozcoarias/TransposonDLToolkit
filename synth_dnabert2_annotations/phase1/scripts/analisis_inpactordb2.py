#!/usr/bin/env python3
"""
Análisis exploratorio de InpactorDB2
Autor: Jorge González Gilbaja
TFM - Detección automática de TEs mediante Deep Learning
"""

import re
import argparse
from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_header(header):
    """
    Parsear el header del FASTA para extraer información relevante.
    Formato esperado: >rnd-X_family-Y_Z#CLASSI/SUPERFAMILY/FAMILY Species
    """
    info = {
        'full_header': header,
        'rnd_id': None,
        'family_id': None,
        'copy_number': None,
        'classification': None,
        'order': None,
        'superfamily': None,
        'family': None,
        'species': None
    }
    
    # Extraer rnd-X
    rnd_match = re.search(r'rnd-(\d+)', header)
    if rnd_match:
        info['rnd_id'] = int(rnd_match.group(1))
    
    # Extraer family-Y
    family_match = re.search(r'family-(\d+)', header)
    if family_match:
        info['family_id'] = int(family_match.group(1))
    
    # Extraer copy number (_Z)
    copy_match = re.search(r'_(\d+)#', header)
    if copy_match:
        info['copy_number'] = int(copy_match.group(1))
    
    # Extraer clasificación (después de #)
    class_match = re.search(r'#([^\s]+)', header)
    if class_match:
        classification = class_match.group(1)
        info['classification'] = classification
        
        # Dividir por /
        parts = classification.split('/')
        if len(parts) >= 1:
            info['order'] = parts[0]  # CLASSI
        if len(parts) >= 2:
            info['superfamily'] = parts[1]  # LTR, LINE, SINE, etc.
        if len(parts) >= 3:
            info['family'] = parts[2]  # GYPSY, COPIA, etc.
    
    # Extraer especie (último elemento después de espacio)
    species_match = re.search(r'\s+(\S+)$', header)
    if species_match:
        info['species'] = species_match.group(1)
    
    return info

def analyze_fasta(fasta_file):
    """
    Analizar un archivo FASTA de InpactorDB2
    """
    sequences = []
    lengths = []
    gc_contents = []
    parsed_info = []
    
    print(f"Analizando {fasta_file}...")
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        length = len(seq)
        
        # Calcular contenido GC
        gc = (seq.upper().count('G') + seq.upper().count('C')) / length * 100
        
        # Parsear header
        info = parse_header(record.description)
        info['length'] = length
        info['gc_content'] = gc
        info['seq_id'] = record.id
        
        sequences.append(seq)
        lengths.append(length)
        gc_contents.append(gc)
        parsed_info.append(info)
    
    df = pd.DataFrame(parsed_info)
    
    return df, sequences

def analyze_fasta_chunked(fasta_file, chunk_size=1000):
    """
    Analizar archivo FASTA en lotes para evitar problemas de memoria
    Útil para archivos muy grandes (>500MB)
    """
    parsed_info = []
    sequences = []
    
    print(f"Analizando {fasta_file} en lotes de {chunk_size}...")
    
    chunk = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        chunk.append(record)
        
        if len(chunk) >= chunk_size:
            # Procesar chunk
            for rec in chunk:
                seq = str(rec.seq)
                length = len(seq)
                gc = (seq.upper().count('G') + seq.upper().count('C')) / length * 100
                
                info = parse_header(rec.description)
                info['length'] = length
                info['gc_content'] = gc
                info['seq_id'] = rec.id
                
                parsed_info.append(info)
                sequences.append(seq)
            
            chunk = []
            print(f"  Procesadas {len(parsed_info)} secuencias...")
    
    # Procesar último chunk
    for rec in chunk:
        seq = str(rec.seq)
        length = len(seq)
        gc = (seq.upper().count('G') + seq.upper().count('C')) / length * 100
        
        info = parse_header(rec.description)
        info['length'] = length
        info['gc_content'] = gc
        info['seq_id'] = rec.id
        
        parsed_info.append(info)
        sequences.append(seq)
    
    print(f"  Total procesadas: {len(parsed_info)} secuencias")
    
    df = pd.DataFrame(parsed_info)
    return df, sequences

def generate_statistics(df):
    """
    Generar estadísticas descriptivas
    """
    stats = {}
    
    stats['total_sequences'] = len(df)
    stats['unique_families'] = df['family_id'].nunique()
    stats['unique_species'] = df['species'].nunique()
    
    # Estadísticas de longitud
    stats['length_mean'] = df['length'].mean()
    stats['length_median'] = df['length'].median()
    stats['length_min'] = df['length'].min()
    stats['length_max'] = df['length'].max()
    stats['length_std'] = df['length'].std()
    
    # Estadísticas de GC
    stats['gc_mean'] = df['gc_content'].mean()
    stats['gc_median'] = df['gc_content'].median()
    stats['gc_std'] = df['gc_content'].std()
    
    # Distribución por clasificación
    stats['order_distribution'] = df['order'].value_counts().to_dict()
    stats['superfamily_distribution'] = df['superfamily'].value_counts().to_dict()
    stats['family_distribution'] = df['family'].value_counts().to_dict()
    
    return stats

def print_report(stats, df):
    """
    Imprimir reporte de análisis
    """
    print("\n" + "="*70)
    print("REPORTE DE ANÁLISIS - InpactorDB2")
    print("="*70)
    
    print(f"\n📊 ESTADÍSTICAS GENERALES")
    print(f"  Total de secuencias: {stats['total_sequences']}")
    print(f"  Familias únicas: {stats['unique_families']}")
    print(f"  Especies únicas: {stats['unique_species']}")
    
    print(f"\n📏 LONGITUDES")
    print(f"  Media: {stats['length_mean']:.2f} bp")
    print(f"  Mediana: {stats['length_median']:.2f} bp")
    print(f"  Rango: {stats['length_min']} - {stats['length_max']} bp")
    print(f"  Desviación estándar: {stats['length_std']:.2f} bp")
    
    print(f"\n🧬 CONTENIDO GC")
    print(f"  Media: {stats['gc_mean']:.2f}%")
    print(f"  Mediana: {stats['gc_median']:.2f}%")
    print(f"  Desviación estándar: {stats['gc_std']:.2f}%")
    
    print(f"\n🏷️  DISTRIBUCIÓN POR ORDEN (CLASSI)")
    for order, count in sorted(stats['order_distribution'].items(), 
                               key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_sequences']) * 100
        print(f"  {order}: {count} ({percentage:.1f}%)")
    
    print(f"\n🏷️  DISTRIBUCIÓN POR SUPERFAMILIA")
    for superfam, count in sorted(stats['superfamily_distribution'].items(), 
                                  key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / stats['total_sequences']) * 100
        print(f"  {superfam}: {count} ({percentage:.1f}%)")
    
    print(f"\n🏷️  DISTRIBUCIÓN POR FAMILIA (Top 10)")
    for fam, count in sorted(stats['family_distribution'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / stats['total_sequences']) * 100
        print(f"  {fam}: {count} ({percentage:.1f}%)")
    
    print(f"\n🌍 TOP 10 ESPECIES")
    species_counts = df['species'].value_counts().head(10)
    for species, count in species_counts.items():
        percentage = (count / stats['total_sequences']) * 100
        print(f"  {species}: {count} ({percentage:.1f}%)")
    
    print("\n" + "="*70)

def create_visualizations(df, output_prefix='analisis'):
    """
    Crear visualizaciones de los datos
    """
    sns.set_style("whitegrid")
    
    # 1. Distribución de longitudes
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histograma de longitudes
    axes[0, 0].hist(df['length'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Longitud (bp)')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_title('Distribución de Longitudes de TEs')
    axes[0, 0].axvline(df['length'].mean(), color='red', linestyle='--', 
                       label=f'Media: {df["length"].mean():.0f} bp')
    axes[0, 0].legend()
    
    # Histograma de contenido GC
    axes[0, 1].hist(df['gc_content'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Contenido GC (%)')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Contenido GC')
    axes[0, 1].axvline(df['gc_content'].mean(), color='red', linestyle='--',
                       label=f'Media: {df["gc_content"].mean():.1f}%')
    axes[0, 1].legend()
    
    # Distribución por superfamilia
    superfam_counts = df['superfamily'].value_counts().head(10)
    axes[1, 0].barh(range(len(superfam_counts)), superfam_counts.values)
    axes[1, 0].set_yticks(range(len(superfam_counts)))
    axes[1, 0].set_yticklabels(superfam_counts.index)
    axes[1, 0].set_xlabel('Número de secuencias')
    axes[1, 0].set_title('Top 10 Superfamilias')
    axes[1, 0].invert_yaxis()
    
    # Scatter: Longitud vs GC content por superfamilia
    top_superfams = df['superfamily'].value_counts().head(5).index
    df_top = df[df['superfamily'].isin(top_superfams)]
    for superfam in top_superfams:
        subset = df_top[df_top['superfamily'] == superfam]
        axes[1, 1].scatter(subset['length'], subset['gc_content'], 
                          label=superfam, alpha=0.6, s=50)
    axes[1, 1].set_xlabel('Longitud (bp)')
    axes[1, 1].set_ylabel('Contenido GC (%)')
    axes[1, 1].set_title('Longitud vs GC Content (Top 5 superfamilias)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_visualizaciones.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualizaciones guardadas en: {output_prefix}_visualizaciones.png")
    
    plt.close()

def export_to_csv(df, output_file='analisis_inpactordb2.csv'):
    """
    Exportar datos analizados a CSV
    """
    df.to_csv(output_file, index=False)
    print(f"\n💾 Datos exportados a: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Análisis exploratorio de InpactorDB2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python analisis_inpactordb2.py -i inpactordb2.fasta
  python analisis_inpactordb2.py -i inpactordb2.fasta -o resultados_completo --chunked
        """
    )
    parser.add_argument('--input', '-i', 
                       default="/mnt/user-data/uploads/r_1_5_fragmento.fasta",
                       help='Archivo FASTA de InpactorDB2')
    parser.add_argument('--output-prefix', '-o', default='analisis',
                       help='Prefijo para archivos de salida (default: analisis)')
    parser.add_argument('--chunked', action='store_true',
                       help='Procesar en lotes (recomendado para archivos >500MB)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Tamaño de lote si se usa --chunked (default: 1000)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ANÁLISIS EXPLORATORIO DE INPACTORDB2")
    print("="*70)
    print(f"Archivo de entrada: {args.input}")
    print(f"Prefijo de salida: {args.output_prefix}")
    print(f"Modo: {'Chunked' if args.chunked else 'Normal'}")
    print("="*70)
    
    # Analizar según el modo
    if args.chunked:
        df, sequences = analyze_fasta_chunked(args.input, args.chunk_size)
    else:
        df, sequences = analyze_fasta(args.input)
    
    stats = generate_statistics(df)
    print_report(stats, df)
    
    # Crear visualizaciones
    create_visualizations(df, output_prefix=args.output_prefix)
    
    # Exportar a CSV
    export_to_csv(df, f'{args.output_prefix}.csv')
    
    print("\n✅ Análisis completado exitosamente!")
    print("\nArchivos generados:")
    print(f"  - {args.output_prefix}_visualizaciones.png")
    print(f"  - {args.output_prefix}.csv")
    print("\nPróximos pasos recomendados:")
    print("  1. Revisar las visualizaciones generadas")
    print("  2. Definir criterios de selección de familias para el dataset sintético")
    print("  3. Comenzar con T1.2: Pipeline de mutación")
