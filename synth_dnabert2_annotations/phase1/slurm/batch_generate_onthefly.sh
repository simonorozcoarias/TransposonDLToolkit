#!/bin/bash
###############################################################################
# Script maestro mejorado para generación masiva (sin archivos duplicados)
# Autor: Jorge González Gilbaja
# TFM - Detección automática de TEs mediante Deep Learning
# Fecha: 16/10/2025
#
# MEJORA: Usa índice JSON para filtrado on-the-fly
# - Solo 2-3 archivos permanentes (vs ~1,500)
# - Sin duplicación de datos
# - Filtrado rápido en memoria
###############################################################################

set -e
set -u

###############################################################################
# CONFIGURACIÓN
###############################################################################

# Archivos principales
INPACTORDB_FASTA="r.1.5_all.fasta"
SPECIES_INDEX="species_index.json"
SPECIES_SUMMARY="species_index_summary.csv"

# Directorios
OUTPUT_DIR="synthetic_genomes"
LOG_DIR="logs"

# Parámetros de TEgenomeSimulator
MIN_COPIES=5
MIN_MAX_COPIES=5
MAX_MAX_COPIES=2000
MIN_IDENTITY=85
MAX_IDENTITY=95
MIN_SD=5
MAX_SD=15

# Control
DRY_RUN=false
CONTINUE_ON_ERROR=true

###############################################################################
# COLORES
###############################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

###############################################################################
# FUNCIONES
###############################################################################

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

check_dependencies() {
    print_header "VERIFICANDO DEPENDENCIAS"
    
    local all_ok=true
    
    # Python y scripts
    if command -v python3 &> /dev/null; then
        print_success "Python3 encontrado"
    else
        print_error "Python3 no encontrado"
        all_ok=false
    fi
    
    if [ -f "generate_genome_onthefly.py" ]; then
        print_success "generate_genome_onthefly.py encontrado"
    else
        print_error "generate_genome_onthefly.py no encontrado"
        all_ok=false
    fi
    
    # Archivos principales
    if [ -f "$INPACTORDB_FASTA" ]; then
        local size=$(du -h "$INPACTORDB_FASTA" | cut -f1)
        print_success "InpactorDB2: $size"
    else
        print_error "Archivo no encontrado: $INPACTORDB_FASTA"
        all_ok=false
    fi
    
    if [ -f "$SPECIES_INDEX" ]; then
        local size=$(du -h "$SPECIES_INDEX" | cut -f1)
        print_success "Índice de especies: $size"
    else
        print_error "Índice no encontrado: $SPECIES_INDEX"
        print_info "Ejecutar: python3 build_species_index.py ..."
        all_ok=false
    fi
    
    if [ ! "$all_ok" = true ]; then
        print_error "Faltan dependencias. Abortando."
        exit 1
    fi
    
    # Mostrar estadísticas del índice
    local num_species=$(python3 -c "import json; print(len(json.load(open('$SPECIES_INDEX'))))")
    print_info "Especies en índice: $num_species"
}

create_directories() {
    print_header "CREANDO DIRECTORIOS"
    
    mkdir -p "$OUTPUT_DIR"
    print_success "Output: $OUTPUT_DIR"
    
    mkdir -p "$LOG_DIR"
    print_success "Logs: $LOG_DIR"
}

get_species_list() {
    print_header "EXTRAYENDO LISTA DE ESPECIES"
    
    # Extraer especies del índice JSON
    python3 -c "
import json
with open('$SPECIES_INDEX') as f:
    index = json.load(f)
for species in sorted(index.keys()):
    print(species)
" > species_to_process.txt
    
    local total=$(wc -l < species_to_process.txt)
    print_success "Especies a procesar: $total"
}

generate_genome() {
    local species=$1
    local current=$2
    local total=$3
    
    print_header "[$current/$total] GENERANDO: $species"
    
    # Número aleatorio de copias
    local max_copies=$((MIN_MAX_COPIES + RANDOM % (MAX_MAX_COPIES - MIN_MAX_COPIES + 1)))
    print_info "Parámetro -m (max copies): $max_copies"
    
    # Directorio de salida
    local species_output_dir="$OUTPUT_DIR/${species}"
    mkdir -p "$species_output_dir"
    
    # Log file
    local log_file="$LOG_DIR/${species}_generation.log"
    
    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN - No se ejecuta realmente"
        return 0
    fi
    
    # Ejecutar con filtrado on-the-fly
    print_info "Filtrando y generando..."
    local start_time=$(date +%s)
    
    if python3 generate_genome_onthefly.py \
        --species "$species" \
        --input-fasta "$INPACTORDB_FASTA" \
        --species-index "$SPECIES_INDEX" \
        --output-dir "$species_output_dir" \
        --max-copies $max_copies \
        --min-copies $MIN_COPIES \
        --min-identity $MIN_IDENTITY \
        --max-identity $MAX_IDENTITY \
        > "$log_file" 2>&1; then
        
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        local minutes=$((elapsed / 60))
        local seconds=$((elapsed % 60))
        
        print_success "Completado en ${minutes}m ${seconds}s"
        
        # Verificar outputs
        local genome_file="$species_output_dir/${species}_synth_genome_sequence_out_final.fasta"
        local gff_file="$species_output_dir/${species}_synth_repeat_annotation_out_final.gff"
        
        if [ -f "$genome_file" ] && [ -f "$gff_file" ]; then
            local genome_size=$(grep -v ">" "$genome_file" | tr -d '\n' | wc -c)
            local num_tes=$(grep -v "^#" "$gff_file" | wc -l)
            
            print_success "Genoma: $genome_size bp | TEs: $num_tes"
            
            echo "$species,$max_copies,$elapsed,$genome_size,$num_tes,success" >> "$OUTPUT_DIR/generation_metadata.csv"
        else
            print_error "Archivos de salida no encontrados"
            echo "$species,$max_copies,$elapsed,0,0,missing_files" >> "$OUTPUT_DIR/generation_metadata.csv"
            return 1
        fi
    else
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        
        print_error "Falló después de ${elapsed}s"
        print_error "Ver log: $log_file"
        
        echo "$species,$max_copies,$elapsed,0,0,failed" >> "$OUTPUT_DIR/generation_metadata.csv"
        return 1
    fi
    
    return 0
}

generate_all_genomes() {
    print_header "GENERACIÓN MASIVA CON FILTRADO ON-THE-FLY"
    
    local total=$(wc -l < species_to_process.txt)
    print_info "Total especies: $total"
    print_info "Estrategia: Filtrado on-the-fly (sin archivos duplicados)"
    
    # Crear metadata file
    echo "species,max_copies_param,time_seconds,genome_size,num_tes,status" > "$OUTPUT_DIR/generation_metadata.csv"
    
    local current=0
    local success=0
    local failed=0
    local start_total=$(date +%s)
    
    while IFS= read -r species; do
        [ -z "$species" ] && continue
        
        current=$((current + 1))
        
        if generate_genome "$species" "$current" "$total"; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
            
            if [ "$CONTINUE_ON_ERROR" = false ]; then
                print_error "Abortando"
                break
            fi
        fi
        
        echo ""
        
    done < species_to_process.txt
    
    # Reporte final
    local end_total=$(date +%s)
    local elapsed_total=$((end_total - start_total))
    local hours=$((elapsed_total / 3600))
    local minutes=$(((elapsed_total % 3600) / 60))
    
    print_header "REPORTE FINAL"
    echo "Procesadas: $current"
    print_success "Exitosas: $success"
    print_error "Fallidas: $failed"
    echo "Tiempo total: ${hours}h ${minutes}m"
    echo ""
    print_success "Metadata: $OUTPUT_DIR/generation_metadata.csv"
}

generate_summary() {
    print_header "GENERANDO RESUMEN"
    
    local metadata="$OUTPUT_DIR/generation_metadata.csv"
    
    if [ ! -f "$metadata" ]; then
        print_warning "Metadata no encontrado"
        return
    fi
    
    local total=$(tail -n +2 "$metadata" | wc -l)
    local success=$(grep "success" "$metadata" | wc -l)
    local failed=$(grep -E "failed|missing" "$metadata" | wc -l)
    
    # Promedios
    local avg_time=$(awk -F',' 'NR>1 && $6=="success" {sum+=$3; count++} END {if(count>0) print int(sum/count); else print 0}' "$metadata")
    local avg_tes=$(awk -F',' 'NR>1 && $6=="success" {sum+=$5; count++} END {if(count>0) print int(sum/count); else print 0}' "$metadata")
    
    # Guardar resumen
    local summary="$OUTPUT_DIR/SUMMARY.txt"
    {
        echo "RESUMEN DE GENERACIÓN (ON-THE-FLY)"
        echo "======================================"
        echo ""
        echo "Fecha: $(date)"
        echo ""
        echo "ESTRATEGIA MEJORADA:"
        echo "  - Filtrado on-the-fly (sin duplicar archivos)"
        echo "  - Solo 2-3 archivos permanentes"
        echo "  - Archivos temporales creados y eliminados automáticamente"
        echo ""
        echo "Total procesadas: $total"
        echo "Exitosas: $success"
        echo "Fallidas: $failed"
        echo ""
        echo "Tiempo promedio: ${avg_time}s"
        echo "TEs promedio: $avg_tes"
        echo ""
        echo "ARCHIVOS PERMANENTES:"
        echo "  1. $INPACTORDB_FASTA (original)"
        echo "  2. $SPECIES_INDEX"
        echo "  3. $SPECIES_SUMMARY"
    } > "$summary"
    
    print_success "Resumen: $summary"
    cat "$summary"
}

###############################################################################
# MAIN
###############################################################################

main() {
    print_header "GENERACIÓN MASIVA MEJORADA (ON-THE-FLY)"
    echo "Fecha: $(date)"
    echo ""
    
    # Argumentos
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                print_warning "Modo DRY RUN"
                shift
                ;;
            --stop-on-error)
                CONTINUE_ON_ERROR=false
                shift
                ;;
            *)
                echo "Argumento desconocido: $1"
                exit 1
                ;;
        esac
    done
    
    check_dependencies
    create_directories
    get_species_list
    generate_all_genomes
    generate_summary
    
    print_header "✅ PROCESO COMPLETADO"
    print_info "Archivos permanentes: 3 (vs ~1,500 con método anterior)"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
