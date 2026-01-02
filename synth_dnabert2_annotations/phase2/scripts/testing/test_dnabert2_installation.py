#!/usr/bin/env python3
"""
Test de instalación de DNABERT-2
Verifica que todas las dependencias estén correctamente instaladas
"""

import sys

def test_imports():
    """Verifica que los paquetes necesarios se puedan importar"""
    print("="*80)
    print("TEST 1: Verificando imports de dependencias...")
    print("="*80)
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'numpy': 'NumPy',
        'triton': 'Triton (opcional)'
    }
    
    failed = []
    
    for package, name in required_packages.items():
        try:
            if package == 'torch':
                import torch
                version = torch.__version__
                cuda_available = torch.cuda.is_available()
                print(f"✓ {name}: v{version}")
                print(f"  - CUDA disponible: {cuda_available}")
                if cuda_available:
                    print(f"  - Dispositivos CUDA: {torch.cuda.device_count()}")
                    print(f"  - GPU actual: {torch.cuda.get_device_name(0)}")
            elif package == 'transformers':
                import transformers
                version = transformers.__version__
                print(f"✓ {name}: v{version}")
            elif package == 'numpy':
                import numpy
                version = numpy.__version__
                print(f"✓ {name}: v{version}")
            elif package == 'triton':
                try:
                    import triton
                    print(f"✓ {name}: instalado (opcional)")
                except ImportError:
                    print(f"⚠ {name}: no instalado (es opcional, no es problema)")
        except ImportError as e:
            print(f"✗ {name}: NO INSTALADO")
            if package != 'triton':  # Triton es opcional
                failed.append(package)
    
    print()
    return len(failed) == 0

def test_dnabert2_model():
    """Intenta cargar el modelo DNABERT-2 desde HuggingFace"""
    print("="*80)
    print("TEST 2: Cargando modelo DNABERT-2 desde HuggingFace...")
    print("="*80)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        model_name = "zhihan1996/DNABERT-2-117M"
        
        print(f"Descargando tokenizer de {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✓ Tokenizer cargado correctamente")
        
        print(f"\nDescargando modelo de {model_name}...")
        print("(Esto puede tardar varios minutos la primera vez)")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print("✓ Modelo cargado correctamente")
        
        # Obtener información del modelo
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n  - Parámetros totales: {total_params:,}")
        print(f"  - Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return True, model, tokenizer
        
    except Exception as e:
        print(f"✗ Error al cargar modelo: {e}")
        return False, None, None

def test_inference():
    """Prueba de inferencia básica con DNABERT-2"""
    print("\n" + "="*80)
    print("TEST 3: Prueba de inferencia con secuencia de DNA...")
    print("="*80)

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        model_name = "zhihan1996/DNABERT-2-117M"

        # Determinar dispositivo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {device}")

        # Cargar modelo y tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Mover modelo al dispositivo apropiado
        model = model.to(device)
        print(f"✓ Modelo movido a {device}")

        # Secuencia de prueba
        test_sequence = "ACGTACGTACGTACGTACGTACGTACGTACGT"
        print(f"Secuencia de prueba: {test_sequence}")

        # Tokenizar
        inputs = tokenizer(test_sequence, return_tensors='pt')
        print(f"✓ Secuencia tokenizada")
        print(f"  - Shape de input_ids: {inputs['input_ids'].shape}")

        # Mover inputs al mismo dispositivo que el modelo
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inferencia
        with torch.no_grad():
            outputs = model(**inputs)

        print(f"✓ Inferencia completada")

        # DNABERT-2 puede devolver tupla o objeto con atributos
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            print(f"  - Tipo de output: tuple")
            print(f"  - Shape del output: {hidden_states.shape}")
            print(f"  - Dimensión de embedding: {hidden_states.shape[-1]}")
        else:
            print(f"  - Tipo de output: {type(outputs)}")
            print(f"  - Shape del output: {outputs.last_hidden_state.shape}")
            print(f"  - Dimensión de embedding: {outputs.last_hidden_state.shape[-1]}")

        return True

    except Exception as e:
        print(f"✗ Error en inferencia: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu():
    """Prueba específica de GPU si está disponible"""
    print("\n" + "="*80)
    print("TEST 4: Verificando capacidad GPU...")
    print("="*80)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠ GPU no disponible - se usará CPU (más lento)")
            return True
        
        print(f"✓ GPU disponible")
        print(f"  - Dispositivos: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Información de memoria
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    Memoria total: {total_memory:.2f} GB")
        
        # Test simple de operación en GPU
        print("\nPrueba de operación en GPU...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print("✓ Operaciones en GPU funcionando correctamente")
        
        return True
        
    except Exception as e:
        print(f"✗ Error en test de GPU: {e}")
        return False

def main():
    print("\n" + "🧬"*40)
    print("VERIFICACIÓN DE INSTALACIÓN DE DNABERT-2")
    print("🧬"*40 + "\n")
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Cargar modelo
    success, model, tokenizer = test_dnabert2_model()
    results.append(("Carga de modelo", success))
    
    # Test 3: Inferencia (solo si el modelo se cargó)
    if success:
        results.append(("Inferencia", test_inference()))
    else:
        print("\n⚠ Saltando test de inferencia (modelo no cargado)")
        results.append(("Inferencia", False))
    
    # Test 4: GPU
    results.append(("GPU", test_gpu()))
    
    # Resumen
    print("\n" + "="*80)
    print("RESUMEN DE TESTS")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_critical_passed = results[0][1] and results[1][1] and results[2][1]
    
    print("\n" + "="*80)
    if all_critical_passed:
        print("✓✓✓ INSTALACIÓN EXITOSA ✓✓✓")
        print("DNABERT-2 está listo para usar")
    else:
        print("✗✗✗ PROBLEMAS DETECTADOS ✗✗✗")
        print("Revisa los errores arriba y reinstala los paquetes necesarios")
    print("="*80 + "\n")
    
    return 0 if all_critical_passed else 1

if __name__ == "__main__":
    sys.exit(main())
