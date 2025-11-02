import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS, add_constant

# Try multiple possible import paths for causal-learn / causallearn to support different package versions
try:
    # Preferred package layout (causal-learn)
    from causal_learn.search.ConstraintBased.PC import pc
    from causal_learn.utils.cit import fisherz
except Exception:
    try:
        # Alternative package name / layout (causallearn)
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
    except Exception as e:
        raise ImportError(
            "Could not import 'pc' and 'fisherz' from causal-learn/causallearn; "
            "please install 'causal-learn' (pip install causal-learn) "
            "and verify the package version and import paths."
        ) from e

# ==============================================================================
# 0. CONFIGURACIÓN Y CONSTANTES 
# ==============================================================================
FILE_PATH = "C:\\GIT\\transport-ml\\data\\raw\\Viajes Sep-Dic 24 v2.xlsx"  # IMPORTANTE: Reemplaza con tu ruta de archivo
TARGET_VARIABLE_NAME = 'CostoxTn'                 
COEFFICIENT_THRESHOLD = 0.05                   
SELECTED_COLUMNS = [
    'TpoTrn.APT', 
    'Variación', 
    'Costo', 
    'CostoxTn', 
    'Flete Falso (MXN)' 
]
EXCEL_SHEET_NAME = "Viajes"

# ==============================================================================
# 1. FUNCIÓN DE VISUALIZACIÓN PARA PC
# ==============================================================================
def plot_pc_dag(adj_matrix, columns, causal_strengths=None, title='Grafo Causal (CPDAG) Algoritmo PC'):
    """Visualiza el grafo de estructura (CPDAG) generado por el Algoritmo PC."""
    G = nx.DiGraph()
    G.add_nodes_from(columns)
    edge_labels = {}
    
    # Mapear la matriz de adyacencia de PC a un grafo NetworkX
    for i, col_i in enumerate(columns):
        for j, col_j in enumerate(columns):
            # El Algoritmo PC usa 1 para i -> j y 3 para i <- j (y 2 para i -- j)
            if adj_matrix[i, j] == 1:
                # i -> j
                G.add_edge(col_i, col_j)
                if causal_strengths and (col_i, col_j) in causal_strengths:
                    weight = causal_strengths[(col_i, col_j)]
                    edge_labels[(col_i, col_j)] = f"{weight:.4f}"
            elif adj_matrix[i, j] == 2 and adj_matrix[j, i] == 2:
                # i -- j (Borde incierto, bidireccional para visualización)
                if not G.has_edge(col_j, col_i):  # Evitar duplicados
                    G.add_edge(col_i, col_j, weight='?')
    
    # Configuración de la visualización
    plt.figure(figsize=(12, 7))
    
    # Intentar usar Graphviz, pero con mejor manejo de excepciones
    pos = None
    try:
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
        print("✓ Layout con Graphviz (dot) exitoso.")
    except Exception as e:
        print(f"⚠️  No se pudo usar Graphviz: {type(e).__name__}. Usando spring_layout...")
        pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_color='lightcoral', node_size=1500, edgecolors='gray')
    
    # Dibuja aristas dirigidas (flechas)
    directed_edges = [(u, v) for u, v in G.edges() if G.edges[u, v].get('weight') != '?']
    nx.draw_networkx_edges(G, pos, edgelist=directed_edges, arrowstyle='->', 
                          arrowsize=20, edge_color='black', width=1.5)
    
    # Dibuja aristas inciertas (baja opacidad)
    undirected_edges = [(u, v) for u, v in G.edges() if G.edges[u, v].get('weight') == '?']
    nx.draw_networkx_edges(G, pos, edgelist=undirected_edges, arrowstyle='-', 
                          style='dashed', edge_color='gray', width=1.0, alpha=0.5)
    
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    # Dibuja las etiquetas de fuerza causal
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                    font_color='red', font_size=8, label_pos=0.5)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 2. BLOQUE DE EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == '__main__':
    # --- 1. Carga, Filtrado y Preprocesamiento de Datos ---
    print("--- 1. Carga, Filtrado y Preprocesamiento de Datos ---")
    try:
        data = pd.read_excel(FILE_PATH, sheet_name="Viajes")
        data = data[SELECTED_COLUMNS]
        data = data.dropna()
        
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data = data.dropna().astype(float)
        print(f"✅ Datos cargados y filtrados. Dimensiones finales: {data.shape}")
        print(f"   Columnas: {list(data.columns)}")
        
    except FileNotFoundError:
        print(f"❌ ERROR: Archivo no encontrado en {FILE_PATH}")
        exit()
    except Exception as e:
        print(f"❌ ERROR durante la carga o limpieza de datos: {e}")
        exit()
    
    # --- 2. Estandarización de Datos ---
    scaler = StandardScaler()
    data_scaled_np = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled_np, columns=data.columns)
    print("\n[Estandarización de Datos]")
    print("✅ Datos escalados con StandardScaler.")
    
    # --- 3. DESCUBRIMIENTO CAUSAL (Algoritmo PC) ---
    print("\n--- 3. Descubrimiento Causal (Algoritmo PC) ---")
    try:
        data_matrix = data_scaled.values
        
        # Ejecutar el Algoritmo PC
        graph_pc = pc(data_matrix, alpha=0.05, indep_test=fisherz)
        
        # CORRECCIÓN: Acceso correcto a la matriz de adyacencia
        adj_matrix = graph_pc.G.graph
        
        print(f"✅ Descubrimiento de Estructura exitoso con Algoritmo PC.")
        print(f"   Variables analizadas: {len(data.columns)}")
        print(f"\nMatriz de Adyacencia (forma: {adj_matrix.shape}):")
        print(adj_matrix)
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO en el Descubrimiento Causal PC: {e}")
        import traceback
        traceback.print_exc()
        exit()
    
    # --- 4. CÁLCULO DE FUERZA CAUSAL (Regresión OLS) ---
    print("\n--- 4. CÁLCULO DE FUERZA CAUSAL (Regresión OLS) ---")
    causal_strengths = {}
    
    # Iterar sobre cada variable como TARGET
    for target_name in data_scaled.columns:
        target_index = data_scaled.columns.get_loc(target_name)
        
        # Identificar las variables PARENTES (causas) con flechas dirigidas al TARGET
        parents_indices = [i for i, col in enumerate(data_scaled.columns) 
                          if adj_matrix[i, target_index] == 1]
        parent_names = [data_scaled.columns[i] for i in parents_indices]
        
        if parent_names:
            # Construir el modelo de regresión
            Y = data_scaled[target_name]
            X = data_scaled[parent_names]
            X = add_constant(X)
            
            try:
                model = OLS(Y, X).fit()
                print(f"\n📊 Modelo para la variable DEPENDIENTE: {target_name}")
                print(f"   R-squared: {model.rsquared:.4f}")
                
                # Recorrer los resultados para obtener los coeficientes
                for parent in parent_names:
                    coefficient = model.params.get(parent, 0.0)
                    p_value = model.pvalues.get(parent, 1.0)
                    
                    # Usar el coeficiente solo si es estadísticamente significativo
                    if p_value < 0.05: 
                        causal_strengths[(parent, target_name)] = coefficient
                        print(f"  ✓ {parent} → {target_name} | CS: {coefficient:.4f} (p={p_value:.4f})")
                    else:
                        print(f"  ✗ {parent} → {target_name} | CS: {coefficient:.4f} (p={p_value:.4f}) [NO SIG]")
                        
            except Exception as e:
                print(f"  ❌ Error en regresión para {target_name}: {e}")
    
    if not causal_strengths:
        print("\n⚠️  ADVERTENCIA: No se encontraron relaciones causales significativas.")
    else:
        print(f"\n✅ Total de relaciones causales significativas encontradas: {len(causal_strengths)}")
    
    # --- 5. VISUALIZACIÓN DEL DAG con Fuerza Causal ---
    print("\n--- 5. Visualización del Grafo Causal ---")
    plot_pc_dag(
        adj_matrix, 
        data_scaled.columns.tolist(), 
        causal_strengths=causal_strengths,
        title=f'Grafo Causal PC + Fuerza OLS (Alpha=0.05)'
    )