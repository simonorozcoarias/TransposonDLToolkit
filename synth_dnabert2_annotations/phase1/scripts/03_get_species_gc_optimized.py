#!/usr/bin/env python3
"""
Obtener %GC óptimo por especie desde NCBI
Autor: Jorge González Gilbaja
TFM - Detección automática de TEs mediante Deep Learning
Fecha: 14/10/2025

Estrategia según tutor:
1. Buscar genoma de referencia (RefSeq)
2. Si no existe, usar ensamblaje con mayor N50
3. Extraer %GC del Assembly Stats
4. Identificar tipo de organismo (animal, planta, hongo, bacteria, etc.)
   mediante consulta de taxonomía NCBI
"""

import argparse
import subprocess
import csv
import sys
import os
import shutil
import re
from typing import Optional, Dict, List
import time

TARGET_INSERT_SIZE = 50_000_000

def find_ncbi_executable(executable_name: str) -> str:
    """
    Buscar un ejecutable de NCBI (datasets o dataformat) en el PATH o en conda envs.
    """
    # Primero intentar encontrar en PATH
    exe_path = shutil.which(executable_name)
    if exe_path:
        return exe_path

    # Si no está en PATH, buscar en conda envs
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        possible_path = os.path.join(conda_prefix, 'bin', executable_name)
        if os.path.exists(possible_path):
            return possible_path

    # Buscar en el entorno ncbi_datasets específico
    home = os.path.expanduser('~')
    ncbi_exe_path = os.path.join(home, 'miniconda3', 'envs', 'ncbi_datasets', 'bin', executable_name)
    if os.path.exists(ncbi_exe_path):
        return ncbi_exe_path

    # Si no se encuentra, retornar el nombre y que falle con mensaje claro
    return executable_name

# Detectar los ejecutables de NCBI al inicio
DATASETS_EXECUTABLE = find_ncbi_executable('datasets')
DATAFORMAT_EXECUTABLE = find_ncbi_executable('dataformat')

def parse_ambiguous_taxon_error(stderr_text: str) -> Optional[List[str]]:
    """
    Parsear el mensaje de error de nombres ambiguos y extraer TODOS los taxids.

    Ejemplo de error:
    Error: The taxonomy name 'Olea' is an exact match for more than one taxid. Please select one of these taxids:
    Olea (genus, taxid: 1522362)
    Olea (genus, taxid: 4145)

    Returns:
        Lista de taxids ordenados de menor a mayor (el más bajo primero), o None si no se encuentran
    """
    if "is an exact match for more than one taxid" not in stderr_text:
        return None

    # Buscar todos los taxids en el texto
    # Patrón: taxid: NÚMERO
    pattern = r'taxid:\s*(\d+)'
    matches = re.findall(pattern, stderr_text)

    if not matches:
        return None

    # Convertir a enteros, ordenar de menor a mayor, y devolver como strings
    taxids = sorted([int(tid) for tid in matches])
    return [str(tid) for tid in taxids]


def run_ncbi_taxonomy_command(species_name: str, verbose: bool = False) -> Optional[Dict]:
    """
    Ejecutar comando de taxonomía para obtener información del reino (Kingdom).
    Equivalente a: datasets summary taxonomy taxon "species" --as-json-lines |
                   dataformat tsv taxonomy --template tax-summary

    Returns:
        Dict con información taxonómica parseada, incluyendo Kingdom
    """
    try:
        # Proceso 1: datasets summary taxonomy
        datasets_process = subprocess.Popen(
            [DATASETS_EXECUTABLE, 'summary', 'taxonomy', 'taxon', species_name, '--as-json-lines'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Proceso 2: dataformat tsv taxonomy
        dataformat_process = subprocess.Popen(
            [DATAFORMAT_EXECUTABLE, 'tsv', 'taxonomy', '--template', 'tax-summary'],
            stdin=datasets_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Permitir que datasets_process reciba SIGPIPE si dataformat_process termina
        datasets_process.stdout.close()

        # Obtener output de dataformat
        stdout, stderr_dataformat = dataformat_process.communicate()

        # Obtener stderr de datasets
        _, stderr_datasets = datasets_process.communicate()

        # Verificar errores
        if dataformat_process.returncode != 0:
            if verbose:
                print(f"    ⚠️  Error en taxonomía: {stderr_dataformat}", file=sys.stderr)
            return None

        if not stdout or not stdout.strip():
            return None

        # Parsear TSV
        lines = stdout.strip().split('\n')
        if len(lines) < 2:  # Necesitamos header + 1 línea de datos
            return None

        # Obtener headers y datos
        headers = lines[0].split('\t')
        values = lines[1].split('\t')

        if len(values) != len(headers):
            return None

        # Crear diccionario con los datos
        tax_data = dict(zip(headers, values))
        return tax_data

    except Exception as e:
        if verbose:
            print(f"    ⚠️  Error obteniendo taxonomía: {e}", file=sys.stderr)
        return None


def get_organism_type(species_name: str = None, verbose: bool = False, tax_data: Optional[Dict] = None) -> str:
    """
    Determinar el tipo de organismo (animal, planta, hongo, otro) basándose en taxonomía NCBI.

    Args:
        species_name: Nombre de la especie (requerido si tax_data es None)
        verbose: Mostrar mensajes informativos
        tax_data: Datos taxonómicos ya obtenidos (opcional). Si se proporciona, se usa directamente.

    Returns:
        String: 'animal', 'planta', 'hongo', 'bacteria', 'archaea', 'virus', 'protista', 'otro'
    """
    # Si no se proporcionaron datos taxonómicos, obtenerlos
    if tax_data is None:
        if species_name is None:
            return 'desconocido'
        tax_data = run_ncbi_taxonomy_command(species_name, verbose=verbose)

    if not tax_data:
        return 'desconocido'

    # Extraer Kingdom y Domain
    kingdom = tax_data.get('Kingdom name', '').strip().lower()
    domain = tax_data.get('Domain/Realm name', '').strip().lower()

    # Mapear a categorías simples
    if kingdom == 'metazoa':
        return 'animal'
    elif kingdom == 'viridiplantae' or kingdom == 'plantae':
        return 'planta'
    elif kingdom == 'fungi':
        return 'hongo'
    elif domain == 'bacteria':
        return 'bacteria'
    elif domain == 'archaea':
        return 'archaea'
    elif domain == 'viruses':
        return 'virus'
    elif kingdom:  # Si hay kingdom pero no coincide con los anteriores
        # Puede ser protista u otros eucariotas
        return 'protista'
    else:
        return 'otro'


def run_ncbi_pipeline_command(species_name: str, verbose: bool = True, tried_taxids: Optional[List[str]] = None) -> Optional[List[Dict]]:
    """
    Ejecutar pipeline completo: datasets | dataformat tsv
    Equivalente a: datasets summary genome taxon "species" --as-json-lines |
                   dataformat tsv genome --fields accession,organism-name,assmstats-gc-percent

    Maneja automáticamente el caso de nombres ambiguos probando todos los taxids en orden (de menor a mayor).
    También obtiene datos taxonómicos (Kingdom, Domain) y los incluye en cada resultado.

    Args:
        species_name: Nombre de la especie o taxid
        verbose: Mostrar mensajes informativos
        tried_taxids: Lista de taxids ya probados (para evitar bucles infinitos)

    Returns:
        Lista de diccionarios con los datos parseados del TSV, incluyendo datos taxonómicos
        (Kingdom name, Domain/Realm name, etc.) en cada elemento.
    """
    if tried_taxids is None:
        tried_taxids = []

    try:
        # Proceso 1: datasets summary genome
        datasets_process = subprocess.Popen(
            [DATASETS_EXECUTABLE, 'summary', 'genome', 'taxon', species_name, '--as-json-lines'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Proceso 2: dataformat tsv genome (recibe input del anterior)
        dataformat_process = subprocess.Popen(
            [DATAFORMAT_EXECUTABLE, 'tsv', 'genome',
             '--fields', 'accession,organism-name,assmstats-gc-percent,assmstats-contig-n50,assminfo-refseq-category,assminfo-name,assminfo-release-date,assminfo-submitter,assminfo-status,assminfo-refseq-category'],
            stdin=datasets_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Permitir que datasets_process reciba SIGPIPE si dataformat_process termina
        datasets_process.stdout.close()

        # Obtener output de dataformat
        stdout, stderr_dataformat = dataformat_process.communicate()

        # Obtener stderr de datasets
        _, stderr_datasets = datasets_process.communicate()

        # Verificar si hubo error de nombre ambiguo en datasets
        if stderr_datasets and "is an exact match for more than one taxid" in stderr_datasets:
            # Extraer TODOS los taxids
            taxids = parse_ambiguous_taxon_error(stderr_datasets)

            if taxids:
                if verbose:
                    print(f"    ℹ️  Nombre ambiguo detectado, encontrados {len(taxids)} taxids: {', '.join(taxids)}")
                    print(f"    🔄 Probando taxids en orden (de menor a mayor)...")

                # Probar cada taxid en orden hasta encontrar uno que funcione
                for taxid in taxids:
                    if taxid in tried_taxids:
                        continue  # Ya se probó este taxid

                    if verbose:
                        print(f"    🧪 Probando taxid: {taxid}")

                    # Intentar con este taxid
                    result = run_ncbi_pipeline_command(taxid, verbose=False, tried_taxids=tried_taxids + [taxid])

                    if result:  # Si este taxid devuelve resultados, usarlo
                        if verbose:
                            print(f"    ✅ Taxid {taxid} proporcionó resultados")
                        return result
                    else:
                        if verbose:
                            print(f"    ⚠️  Taxid {taxid} no proporcionó resultados, probando siguiente...")

                # Si ningún taxid funcionó
                if verbose:
                    print(f"    ❌ Ninguno de los {len(taxids)} taxids proporcionó resultados", file=sys.stderr)
                return None
            else:
                if verbose:
                    print(f"    ⚠️  No se pudo resolver nombre ambiguo", file=sys.stderr)
                return None

        # Verificar errores en dataformat
        if dataformat_process.returncode != 0:
            if verbose:
                print(f"    ❌ Error en dataformat: {stderr_dataformat}", file=sys.stderr)
            return None

        if not stdout or not stdout.strip():
            return None

        # Parsear TSV
        lines = stdout.strip().split('\n')
        if len(lines) < 2:  # Necesitamos al menos header + 1 línea de datos
            return None

        # Obtener headers
        headers = lines[0].split('\t')

        # Parsear datos
        results = []
        for line in lines[1:]:
            values = line.split('\t')
            if len(values) == len(headers):
                row_dict = dict(zip(headers, values))
                results.append(row_dict)

        if not results:
            return None

        # Obtener datos taxonómicos para la misma especie/taxid
        tax_data = run_ncbi_taxonomy_command(species_name, verbose=False)

        # Si se obtuvieron datos taxonómicos, agregarlos a cada resultado
        if tax_data:
            for result in results:
                result.update(tax_data)

        return results

    except FileNotFoundError as e:
        print(f"  ❌ ERROR: Comando no encontrado: {e}", file=sys.stderr)
        print("  Instalar con: conda install -c conda-forge ncbi-datasets-cli", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if verbose:
            print(f"  ❌ Error ejecutando pipeline: {e}", file=sys.stderr)
        return None

def get_gc_for_species(species_name: str, verbose: bool = True) -> Optional[Dict]:
    """
    Obtener %GC óptimo para una especie según estrategia del tutor:
    1. Buscar genoma de referencia (RefSeq)
    2. Si no existe, usar ensamblaje con mayor N50
    3. Determinar tipo de organismo (animal, planta, hongo, etc.)

    Usa el pipeline: datasets summary genome taxon | dataformat tsv genome

    Returns:
        Dict con: {
            'species': str,
            'accession': str,
            'gc_percent': float,
            'source': str ('refseq' o 'best_n50'),
            'assembly_name': str,
            'n50': int,
            'organism_type': str ('animal', 'planta', 'hongo', etc.)
        }
    """
    if verbose:
        print(f"  🔍 Buscando: {species_name}")

    # Ejecutar pipeline datasets | dataformat (incluye datos taxonómicos)
    result = run_ncbi_pipeline_command(species_name)

    if not result or len(result) == 0:
        if verbose:
            print(f"    ⚠️  No se encontraron ensamblajes")
        return None

    # Extraer organism_type de los datos taxonómicos incluidos en el primer resultado
    # (todos los resultados tienen los mismos datos taxonómicos)
    organism_type = get_organism_type(tax_data=result[0])
    if verbose and organism_type != 'desconocido':
        print(f"    🧬 Tipo: {organism_type}")

    # Extraer datos de todos los ensamblajes
    assemblies = []

    for row in result:
        try:
            # Datos del TSV (headers definidos en run_ncbi_pipeline_command)
            accession = row.get('Assembly Accession', 'N/A')
            assembly_name = row.get('Assembly Name', 'N/A')

            # %GC - puede venir vacío
            gc_str = row.get('Assembly Stats GC Percent', '').strip()
            if not gc_str or gc_str == 'NA':
                continue
            gc_percent = float(gc_str)

            # N50 - puede venir vacío
            n50_str = row.get('Assembly Stats Contig N50', '0').strip()
            n50 = int(n50_str) if n50_str and n50_str != 'NA' else 0

            # RefSeq category
            refseq_category = row.get('Assembly Refseq Category', '').strip()
            is_refseq = refseq_category == 'reference genome'

            # Release date
            release_date = row.get('Assembly Release Date', '').strip()

            # Submitter
            submitter = row.get('Assembly Submitter', '').strip()

            # Status
            assembly_status = row.get('Assembly Status', '').strip()

            assemblies.append({
                'accession': accession,
                'assembly_name': assembly_name,
                'gc_percent': gc_percent,
                'n50': n50,
                'is_refseq': is_refseq,
                'refseq_category': refseq_category,
                'release_date': release_date,
                'submitter': submitter,
                'status': assembly_status
            })

        except (KeyError, TypeError, ValueError) as e:
            if verbose:
                print(f"    ⚠️  Error parseando ensamblaje: {e}")
            continue

    if not assemblies:
        if verbose:
            print(f"    ⚠️  No se encontró información de GC")
        return None

    # Estrategia 1: Buscar RefSeq
    refseq_assemblies = [a for a in assemblies if a['is_refseq']]

    if refseq_assemblies:
        # Hay genoma(s) de referencia
        # Si hay múltiples, aplicar criterio de mayor N50
        if len(refseq_assemblies) > 1:
            refseq_assemblies.sort(key=lambda x: x['n50'], reverse=True)
            if verbose:
                print(f"    ℹ️  Encontrados {len(refseq_assemblies)} genomas de referencia, seleccionando el de mayor N50")

        best = refseq_assemblies[0]
        if verbose:
            print(f"    ✅ RefSeq: {best['accession']} | GC: {best['gc_percent']:.2f}% | N50: {best['n50']:,}")

        return {
            'species': species_name,
            'accession': best['accession'],
            'assembly_name': best['assembly_name'],
            'gc_percent': best['gc_percent'],
            'n50': best['n50'],
            'source': 'refseq',
            'organism_type': organism_type
        }

    # Estrategia 2: Mayor N50
    assemblies.sort(key=lambda x: x['n50'], reverse=True)
    best = assemblies[0]

    if verbose:
        print(f"    ✅ Mayor N50: {best['accession']} | GC: {best['gc_percent']:.2f}% | N50: {best['n50']:,}")

    return {
        'species': species_name,
        'accession': best['accession'],
        'assembly_name': best['assembly_name'],
        'gc_percent': best['gc_percent'],
        'n50': best['n50'],
        'source': 'best_n50',
        'organism_type': organism_type
    }

def process_species_list(input_csv: str, output_csv: str,
                        max_species: Optional[int] = None,
                        rate_limit_seconds: float = 1.0,
                        verbose: bool = True):
    """
    Procesar lista de especies y obtener %GC para cada una.
    También calcula max/min copies basándose en datos del CSV de entrada.
    """
    # Leer CSV de entrada
    species_list = []

    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            species_list.append(row)

    if max_species:
        species_list = species_list[:max_species]

    total = len(species_list)

    if verbose:
        print("=" * 70)
        print(f"OBTENCIÓN DE %GC PARA {total} ESPECIES")
        print("=" * 70)
        print(f"Archivo entrada: {input_csv}")
        print(f"Archivo salida: {output_csv}")
        print(f"Rate limit: {rate_limit_seconds}s entre consultas")
        print("=" * 70)
    
    # Procesar cada especie
    results = []
    success_count = 0
    failed_species = []
    
    for idx, row in enumerate(species_list, 1):
        species = row['species']
        
        if verbose:
            print(f"\n[{idx}/{total}] Procesando: {species}")
        
        # Obtener %GC
        gc_data = get_gc_for_species(species, verbose=verbose)

        # Calcular max/min copies desde los datos del CSV de entrada
        try:
            num_sequences = int(row.get('num_sequences', 0))
            length_mean = float(row.get('length_mean', 0))

            if num_sequences > 0 and length_mean > 0:
                # Objetivo 50,000,000 bp de insertos de TEs
                # Fórmula base: m_calculado = TARGET_INSERT_SIZE / (n_consensos × longitud_promedio)
                # El factor de escalado para compensar fragmentación se aplica en runtime
                m_calculated = TARGET_INSERT_SIZE / (num_sequences * length_mean)
                max_copies_calculated = int(m_calculated * 1.10)  # +10%
                min_copies_calculated = int(m_calculated * 0.90)  # -10%

                if verbose:
                    print(f"    📊 Calculando copies: {num_sequences} seqs × {length_mean:.1f}bp → max={max_copies_calculated}, min={min_copies_calculated}")
            else:
                max_copies_calculated = None
                min_copies_calculated = None
                if verbose:
                    print(f"    ⚠️  No se pueden calcular copies: datos insuficientes en CSV")
        except (ValueError, KeyError):
            max_copies_calculated = None
            min_copies_calculated = None
            if verbose:
                print(f"    ⚠️  No se pueden calcular copies: datos inválidos en CSV")

        if gc_data:
            # Agregar datos originales del CSV + gc_data + datos calculados
            result = {
                **row,
                **gc_data,
                'max_copies_calculated': max_copies_calculated,
                'min_copies_calculated': min_copies_calculated
            }
            results.append(result)
            success_count += 1
        else:
            # Marcar como fallido, pero intentar obtener al menos el tipo de organismo
            organism_type = get_organism_type(species, verbose=False)
            result = {
                **row,
                'accession': 'N/A',
                'assembly_name': 'N/A',
                'gc_percent': None,
                'n50': None,
                'source': 'failed',
                'organism_type': organism_type,
                'max_copies_calculated': max_copies_calculated,
                'min_copies_calculated': min_copies_calculated
            }
            results.append(result)
            failed_species.append(species)
        
        # Rate limiting para no abrumar NCBI
        if idx < total:
            time.sleep(rate_limit_seconds)
    
    # Guardar resultados
    if results:
        fieldnames = list(results[0].keys())
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    # Reporte final
    if verbose:
        print("\n" + "=" * 70)
        print("📊 REPORTE FINAL")
        print("=" * 70)
        print(f"  Total procesadas: {total}")
        print(f"  ✅ Exitosas: {success_count} ({100*success_count/total:.1f}%)")
        print(f"  ❌ Fallidas: {len(failed_species)} ({100*len(failed_species)/total:.1f}%)")
        
        if failed_species:
            print(f"\n  ⚠️  Especies sin datos:")
            for species in failed_species[:10]:
                print(f"    - {species}")
            if len(failed_species) > 10:
                print(f"    ... y {len(failed_species)-10} más")
        
        # Estadísticas de %GC
        gc_values = [r['gc_percent'] for r in results if r['gc_percent'] is not None]
        if gc_values:
            print(f"\n  🧪 Estadísticas de %GC:")
            print(f"    Media: {sum(gc_values)/len(gc_values):.2f}%")
            print(f"    Mínimo: {min(gc_values):.2f}%")
            print(f"    Máximo: {max(gc_values):.2f}%")
        
        # Estadísticas por source
        refseq_count = sum(1 for r in results if r.get('source') == 'refseq')
        n50_count = sum(1 for r in results if r.get('source') == 'best_n50')

        print(f"\n  📚 Fuentes:")
        print(f"    RefSeq: {refseq_count} ({100*refseq_count/success_count:.1f}%)")
        print(f"    Mayor N50: {n50_count} ({100*n50_count/success_count:.1f}%)")

        # Estadísticas por tipo de organismo
        organism_types = {}
        for r in results:
            org_type = r.get('organism_type', 'desconocido')
            organism_types[org_type] = organism_types.get(org_type, 0) + 1

        if organism_types:
            print(f"\n  🦠 Tipos de organismos:")
            # Ordenar por cantidad (descendente)
            for org_type, count in sorted(organism_types.items(), key=lambda x: x[1], reverse=True):
                print(f"    {org_type.capitalize()}: {count} ({100*count/total:.1f}%)")

        print("\n" + "=" * 70)
        print(f"💾 Resultados guardados en: {output_csv}")
        print("=" * 70)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Obtener %GC óptimo por especie desde NCBI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ESTRATEGIA:
  1. Buscar genoma de referencia (RefSeq)
  2. Si no existe, usar ensamblaje con mayor N50
  3. Extraer %GC del Assembly Stats

PRERREQUISITOS:
  Instalar ncbi-datasets-cli:
    conda install -c conda-forge ncbi-datasets-cli

EJEMPLOS DE USO:
  # Procesar lista completa de especies
  python get_species_gc_optimized.py \\
      --input species_list.csv \\
      --output species_gc_data.csv
  
  # Solo primeras 10 especies (prueba)
  python get_species_gc_optimized.py \\
      --input species_list.csv \\
      --output species_gc_test.csv \\
      --max-species 10
  
  # Con rate limit más agresivo (si NCBI responde bien)
  python get_species_gc_optimized.py \\
      --input species_list.csv \\
      --output species_gc_data.csv \\
      --rate-limit 0.5
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='CSV con lista de especies (output de extract_species)')
    parser.add_argument('--output', '-o', required=True,
                       help='CSV de salida con %GC añadido')
    parser.add_argument('--max-species', type=int,
                       help='Procesar solo las primeras N especies (útil para pruebas)')
    parser.add_argument('--rate-limit', type=float, default=1.0,
                       help='Segundos de espera entre consultas a NCBI (default: 1.0)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Modo silencioso')
    
    args = parser.parse_args()
    
    verbose = not args.quiet

    # Verificar que datasets y dataformat estén instalados
    if verbose:
        print(f"📍 Usando ejecutables:")
        print(f"  - datasets: {DATASETS_EXECUTABLE}")
        print(f"  - dataformat: {DATAFORMAT_EXECUTABLE}")

    # Verificar datasets
    try:
        subprocess.run([DATASETS_EXECUTABLE, '--version'],
                      capture_output=True,
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ERROR: 'datasets' no está instalado o no está en PATH")
        print("\nInstalar con:")
        print("  conda install -c conda-forge ncbi-datasets-cli")
        print("\nO descargar desde:")
        print("  https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/")
        sys.exit(1)

    # Verificar dataformat
    try:
        subprocess.run([DATAFORMAT_EXECUTABLE, 'version'],
                      capture_output=True,
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ERROR: 'dataformat' no está instalado o no está en PATH")
        print("\nInstalar con:")
        print("  conda install -c conda-forge ncbi-datasets-cli")
        print("\nO descargar desde:")
        print("  https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/")
        sys.exit(1)
    
    # Procesar especies
    process_species_list(
        args.input,
        args.output,
        max_species=args.max_species,
        rate_limit_seconds=args.rate_limit,
        verbose=verbose
    )
    
    print("\n✅ Proceso completado exitosamente!")
    print("\n📋 Próximos pasos:")
    print("  1. Revisar species_gc_data.csv")
    print("  2. Verificar especies fallidas (si las hay)")
    print("  3. Proceder con T1.3: Organizar TEs por especie")

if __name__ == "__main__":
    main()
