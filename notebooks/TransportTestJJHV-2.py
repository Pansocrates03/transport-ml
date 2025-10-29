import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import itertools
from copy import deepcopy
from scipy.stats import dirichlet

# --- FIX PGMpy: Alias y uso de la clase moderna 'BayesianNetwork' ---
from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# RUTA AL ARCHIVO DE CONFIGURACIÓN (Asegúrate de que esta ruta sea correcta)
CONFIG_FILE_PATH = 'notebooks/model_parameters.json'
TARGET_VALUE = 1
ROUNDS = 1000
# Lista de solo las dos variables de entrada que queremos analizar sobre X3
INTERVENTION_VARS_TO_ANALYZE = ['X1', 'X2'] 


# ==============================================================================
# 1. CLASE BASE PARA EL MODELO GRÁFICO (BaseModel)
# ==============================================================================

class BaseModel(object):
    def __init__(self, config_file_path=None, data=None):
        self.config_file_path = config_file_path
        self.digraph = None
        self.pgmodel = None
        self.infer_system = None
        self.ebunch = None
        self.nodes = None
        self.variables_dict = dict()
        self.target = None
        self.nature_variables = []
        self.intervention_variables = []
        cpdtables_internal = []
        
        if config_file_path:
            try:
                with open(config_file_path) as json_file:
                    data = json.load(json_file)
            except FileNotFoundError:
                print(f"Error: Archivo de configuración '{config_file_path}' no encontrado.")
                return

        if data.get('model_definition'):
            definition = data['model_definition']
            self.nodes = definition.get('nodes', [])
            self.ebunch = definition.get('digraph', [])
            # Asumiendo X3 es el target
            self.target = definition.get('target', 'X3') 
            self.nature_variables = definition.get('nature_variables', [])
            self.intervention_variables = definition.get('interventions', [])

        if data.get('cpds'):
            for cpd in data['cpds']:
                var_card = int(cpd["cardinality"])
                parent_cards = [int(c) for c in cpd.get("parent_cardinalities", [])]
                
                cpdtables_internal.append({
                    "variable": cpd["variable"],
                    "variable_card": var_card,
                    "values": cpd["probabilities"],
                    "evidence": cpd.get("parents", []),
                    "evidence_card": parent_cards
                })

        if self.ebunch:
            self.pgmodel = BayesianModel(self.ebunch) 
            if self.nodes:
                self.pgmodel.add_nodes_from(self.nodes)
            self.init_graph(ebunch=self.ebunch, nodes=self.nodes)
        
        if cpdtables_internal:
            self.init_model(self.ebunch, cpdtables_internal)
            
        
    def init_graph(self, ebunch, nodes=[], plot=False, graph_id='figures/dag'):
        self.digraph = nx.DiGraph(ebunch)
        for node in nodes:
            self.digraph.add_node(node)
        
    def init_model(self, ebunch, cpdtables, plot=False, pgm_id='pgm'):
        for cpdtable in cpdtables:
            self.variables_dict[cpdtable['variable']] = [_ for _ in range(cpdtable['variable_card'])]
            
            cpd_values = cpdtable['values']
            final_values = cpd_values
            
            if not cpdtable.get('evidence'):
                np_array = np.array(cpd_values, dtype=float).flatten() 
                final_values = np_array.reshape(cpdtable['variable_card'], 1)

            table = TabularCPD(variable=cpdtable['variable'],
                                variable_card=cpdtable['variable_card'],
                                values=final_values, 
                                evidence_card=cpdtable.get('evidence_card'),
                                evidence=cpdtable.get('evidence'))
            
            if cpdtable.get('evidence'):
                table.reorder_parents(sorted(cpdtable.get('evidence'))) 
                
            self.pgmodel.add_cpds(table)
            
        if not self.pgmodel.check_model():
            raise ValueError("Error with CPDTs")
        self.update_infer_system()

    def update_infer_system(self):
        self.infer_system = VariableElimination(self.pgmodel)
        
    def get_graph_toposort(self):
        return list(nx.topological_sort(self.digraph))
        
    def get_nodes_and_predecessors(self):
        return { node : sorted(self.digraph.predecessors(node)) 
                 for node in self.digraph.nodes
               }

    def plot_dag(self, causal_strengths_dict=None, title='DAG con Todas las Fuerzas Causales Aprendidas', node_size=1500, font_size=10):
        """
        Visualiza el DAG y etiqueta las aristas con las fuerzas causales.
        causal_strengths_dict debe contener {'X1_to_X3': cs1, 'X2_to_X3': cs2, 'X1_to_X2': cs3}
        """
        if self.digraph is None:
            print("Error: El grafo no ha sido inicializado.")
            return

        plt.figure(figsize=(8, 5))
        
        try:
            # Layout basado en la estructura causal típica (X1->X2->X3)
            pos = nx.drawing.nx_pydot.graphviz_layout(self.digraph, prog='dot') 
        except:
            pos = nx.spring_layout(self.digraph)

        nx.draw_networkx_nodes(self.digraph, pos, node_color='lightblue', 
                               node_size=node_size, edgecolors='gray')

        nx.draw_networkx_edges(self.digraph, pos, arrowstyle='->', arrowsize=20, 
                               edge_color='black', width=1.5)

        node_labels = {node: node for node in self.digraph.nodes()}
        nx.draw_networkx_labels(self.digraph, pos, labels=node_labels, 
                                font_size=font_size, font_weight='bold')
        
        # --- LÓGICA PARA ETIQUETAR TODAS LAS ARISTAS CON LA FUERZA CAUSAL ---
        if causal_strengths_dict is not None and isinstance(causal_strengths_dict, dict):
            
            edge_labels = {}
            target_var = self.get_target_variable() # X3

            # 1. Etiquetar X2 -> X3
            if 'X2_to_X3' in causal_strengths_dict and self.digraph.has_edge('X2', target_var):
                cs_x2_to_x3 = causal_strengths_dict['X2_to_X3']
                edge_labels[('X2', target_var)] = f"CS(X2->X3): {cs_x2_to_x3:.4f}"

            # 2. Etiquetar X1 -> X3
            if 'X1_to_X3' in causal_strengths_dict and self.digraph.has_edge('X1', target_var):
                cs_x1_to_x3 = causal_strengths_dict['X1_to_X3']
                edge_labels[('X1', target_var)] = f"CS(X1->X3): {cs_x1_to_x3:.4f}" # Usar label_pos si hay superposición

            # 3. Etiquetar X1 -> X2 (El arco intermedio)
            if 'X1_to_X2' in causal_strengths_dict and self.digraph.has_edge('X1', 'X2'):
                cs_x1_to_x2 = causal_strengths_dict['X1_to_X2']
                edge_labels[('X1', 'X2')] = f"CS(X1->X2): {cs_x1_to_x2:.4f}"
            
            # Dibujar todas las etiquetas de aristas
            nx.draw_networkx_edge_labels(
                self.digraph, 
                pos, 
                edge_labels=edge_labels, 
                font_color='red', 
                font_size=10, 
                label_pos=0.5
            )
        # --- FIN LÓGICA FUERZA CAUSAL ---

        plt.title(title)
        plt.axis('off')
        plt.show()

    # --- Métodos Getter (simplificados) ---
    def get_variable_values(self, variable): return self.variables_dict.get(variable)
    def get_target_variable(self): return self.target
    def get_intervention_variables(self): return self.intervention_variables
    def get_nature_variables(self): return self.nature_variables
    
    def get_nature_var_prob(self, nature_variable):
        if nature_variable in self.nature_variables:
            return np.squeeze(self.pgmodel.get_cpds(nature_variable).get_values())

    def conditional_probability(self, variable, evidence):
        return self.infer_system.query([variable], 
                                     evidence=evidence, show_progress=False)

    def make_inference(self, variable, evidence):
        return self.infer_system.map_query([variable],
                                     evidence=evidence, show_progress=False)[variable]


# ==============================================================================
# 2. ENTORNO CAUSAL VERDADERO (TrueCausalModel)
# ==============================================================================

class TrueCausalModel:
    def __init__(self, model):
        self.model = model

    def action_simulator(self, chosen_actions, values_chosen_actions):
        response = dict()
        
        for nat_var in self.model.get_nature_variables():
            probabilities = self.model.get_nature_var_prob(nat_var)
            elements = [i for i in range(len(probabilities))]
            res = np.random.choice(elements, p=probabilities)
            response[nat_var] = res
            
        for idx, variable in enumerate(chosen_actions):
            response[variable] = values_chosen_actions[idx]
            
        ordered_variables = self.model.get_graph_toposort()
        ordered_variables = [i for i in ordered_variables if i not in response]
        
        for unknown_variable in ordered_variables:
            if not response.get(unknown_variable):
                response[unknown_variable] = self.model.make_inference(unknown_variable, response)
                
        return response

# ==============================================================================
# 3. CLASES BASE DE AGENTES (Agent y CausalAgent)
# ==============================================================================

class Agent(object):
    def __init__(self, nature):
        self.nature = nature
        self.rewards_per_round = [0]
        self.n_rounds = 0
    def training(self, rounds): raise NotImplementedError
    def make_decision(self): raise NotImplementedError
    def get_rewards(self): return self.rewards_per_round

class CausalAgent(Agent):
    def __init__(self, nature, pgmodel):
        super().__init__(nature) 
        self.beliefs = dict()
        self.model = deepcopy(pgmodel)
        
    def do_calculus(self, target, intervened_variables):
        return self.model.conditional_probability(target, intervened_variables).values
    
    def make_decision_advanced(self, target, intervened_variables, threshold=-float("inf")):
        target_name = target["variable"]
        target_value = int(target["value"])
        val_inter_vars = [self.model.get_variable_values(i) for i in intervened_variables]
        
        best_actions = None
        max_prob = threshold
        
        for vars_tuples in itertools.product(*val_inter_vars):
            query_dict = dict()
            for i in range(len(intervened_variables)):
                query_dict[intervened_variables[i]] = vars_tuples[i]
            
            prob_table = self.do_calculus(target_name, query_dict)
            prob = prob_table[target_value]
            
            if prob >= max_prob: 
                max_prob = prob
                best_actions = vars_tuples
                
        return best_actions

    def get_causal_strength(self, target_variable, intervention_variable, target_value=1):
        """
        Calcula la Fuerza Causal (Diferencia de Riesgo Absoluto) entre una variable
        intervenida y una variable objetivo.
        """
        # P(Target=1 | do(X_interv=1))
        prob_if_do_1_array = self.do_calculus(
            target_variable, 
            {intervention_variable: 1}
        )
        prob_if_do_1 = prob_if_do_1_array[target_value].item()

        # P(Target=1 | do(X_interv=0))
        prob_if_do_0_array = self.do_calculus(
            target_variable, 
            {intervention_variable: 0}
        )
        prob_if_do_0 = prob_if_do_0_array[target_value].item()
        
        causal_strength = prob_if_do_1 - prob_if_do_0
        
        return causal_strength, prob_if_do_1, prob_if_do_0


# ==============================================================================
# 4. AGENTE DE APRENDIZAJE BAYESIANO (HalfBlindAgent)
# ==============================================================================

class HalfBlindAgent(CausalAgent):
    def __init__(self, nature, pgmodel):
        super().__init__(nature, pgmodel) 
        self.alpha_params = dict()
        self.init_alpha_and_beliefs()

    def init_alpha_and_beliefs(self):
        logging.info("Initializing alpha parameters")
        adj_list = self.model.get_nodes_and_predecessors()
        vars_possible_values = {n : self.model.get_variable_values(n) for n in adj_list}
        
        for node in adj_list:
            node_object = dict()
            combinations = itertools.product(*[vars_possible_values[p] for p in adj_list[node]])
            node_object["has_parents"] = True if len(adj_list[node]) > 0 else False
            
            for combination in combinations:
                parents_to_string = ""
                for i in range(len(combination)):
                    parents_to_string += "{}_{} ".format(adj_list[node][i], combination[i])
                parents_to_string = parents_to_string.strip()
                
                # Inicialización usando np.ones (uniforme)
                k = len(vars_possible_values[node])
                alpha_value = np.ones(k) 
                node_object[parents_to_string] = alpha_value.tolist()
            
            if not node_object["has_parents"]:
                 k = len(vars_possible_values[node])
                 node_object[""] = np.ones(k).tolist()

            self.alpha_params[node] = deepcopy(node_object)
        
        self.update_beliefs()
        self.update_cpts_causal_model()

    def update_beliefs(self, observation_dict=None):
        if observation_dict:
            self.update_alpha_parameters(observation_dict)
            
        for variable in self.alpha_params:
            table = []
            if not self.alpha_params[variable]["has_parents"]:
                alpha = self.alpha_params[variable][""]
                
                probs_array = dirichlet.rvs(alpha, size=1)
                k = len(alpha)
                table = probs_array.reshape(k, 1).tolist() 

            else:
                for parents_instance in self.alpha_params[variable]:
                    if parents_instance == "has_parents":
                        continue
                    alpha = self.alpha_params[variable][parents_instance]
                    probabilities = np.squeeze(dirichlet.rvs(alpha, size=1))
                    table.append(probabilities)
                
                table = np.array(table).transpose().tolist()
            
            self.beliefs[variable] = table

    def update_cpts_causal_model(self):
        adj_list = self.model.get_nodes_and_predecessors()
        var_values = {n : self.model.get_variable_values(n) for n in adj_list}
        
        # Crear un nuevo modelo bayesiano para reemplazar el anterior
        new_pgmodel = BayesianModel(self.model.ebunch) 
        new_pgmodel.add_nodes_from(self.model.nodes)

        for variable in self.beliefs:
            evidence = adj_list[variable]
            evidence_card = [len(var_values[parent]) for parent in evidence]
            
            cpd_table = TabularCPD(variable=variable, 
                                    variable_card=len(var_values[variable]), 
                                    values=self.beliefs[variable],
                                    evidence=evidence, 
                                    evidence_card=evidence_card)
            
            new_pgmodel.add_cpds(cpd_table)

        if new_pgmodel.check_model():
            self.model.pgmodel = new_pgmodel
            self.model.update_infer_system()
        else:
            # Fallback en caso de error
            raise ValueError("Error con CPTs después de la actualización (check_model falló)")

    def update_alpha_parameters(self, observation_dict):
        adj_list = self.model.get_nodes_and_predecessors()
        
        for node in adj_list:
            node_value = observation_dict[node]
            parents_to_string = ""
            
            for parent in adj_list[node]:
                parents_to_string += "{}_{} ".format(parent, observation_dict[parent])
            parents_to_string = parents_to_string.strip()
            
            if not self.alpha_params[node]["has_parents"]:
                 key = ""
            else:
                 key = parents_to_string

            self.alpha_params[node][key][node_value] += 1

    def training(self, rounds, target_value):
        # El agente solo interviene en las variables de intervención definidas en el modelo (usualmente X2)
        intervention_vars = self.model.get_intervention_variables() 
        target = {
            "variable": self.model.get_target_variable(),
            "value" : target_value
            }
            
        for i in range(1, rounds + 1):
            self.model.update_infer_system() 
            
            # Elige la mejor acción sobre las variables de intervención (ej: X2)
            best_actions = self.make_decision_advanced(target, intervention_vars)
            
            # Simula la respuesta de la naturaleza
            nature_response = self.nature.action_simulator(intervention_vars, best_actions)
            
            reward = nature_response[target["variable"]]
            self.rewards_per_round.append(reward)
            
            # Aprende de la observación y actualiza las creencias
            self.update_beliefs(nature_response)
            self.update_cpts_causal_model()
            
            if i % 100 == 0:
                 current_total_reward = sum(self.rewards_per_round)
                 print(f"Ronda {i}: Recompensa Acumulada = {current_total_reward:.2f}")

        final_reward = sum(self.rewards_per_round)
        print(f"\n--- Entrenamiento Finalizado ---")
        print(f"Rondas: {rounds}, Recompensa Total: {final_reward:.2f}")
        
        return self.rewards_per_round

# ==============================================================================
# 5. BLOQUE DE EJECUCIÓN (Cálculo de las 3 Fuerzas Causales Principales)
# ==============================================================================

if __name__ == '__main__':
    
    print(f"--- Análisis Causal Completo (X1, X2, X3) ---")
    print(f"Cargando modelo desde {CONFIG_FILE_PATH}...")
    
    try:
        # 1. Se carga el modelo inicial
        true_model = BaseModel(CONFIG_FILE_PATH)
        
        if true_model.pgmodel:
            nature = TrueCausalModel(true_model)
            half_blind_agent = HalfBlindAgent(nature, true_model)
            
            print("Iniciando entrenamiento del Agente Causal Bayesiano (HalfBlindAgent)...")
            
            # 2. ENTRENAMIENTO
            rewards = half_blind_agent.training(ROUNDS, TARGET_VALUE)

            # 3. ANÁLISIS FINAL: CÁLCULO DE LAS 3 FUERZAS CAUSALES
            results = {} 
            target_var_3 = true_model.get_target_variable() # X3 (Target principal)

            # A. Fuerza Causal 1: do(X2) -> X3
            cs_x2_to_x3, p_x2_1, p_x2_0 = half_blind_agent.get_causal_strength(
                target_var_3, 'X2', TARGET_VALUE
            )
            results['X2_to_X3'] = {'cs': cs_x2_to_x3, 'interv': 'X2', 'target': target_var_3, 'p_do_1': p_x2_1, 'p_do_0': p_x2_0}

            # B. Fuerza Causal 2: do(X1) -> X3 (Efecto total de X1 sobre X3)
            cs_x1_to_x3, p_x1_1, p_x1_0 = half_blind_agent.get_causal_strength(
                target_var_3, 'X1', TARGET_VALUE
            )
            results['X1_to_X3'] = {'cs': cs_x1_to_x3, 'interv': 'X1', 'target': target_var_3, 'p_do_1': p_x1_1, 'p_do_0': p_x1_0}

            # C. Fuerza Causal 3: do(X1) -> X2 (Efecto causal intermedio)
            cs_x1_to_x2, p_x1_1_x2, p_x1_0_x2 = half_blind_agent.get_causal_strength(
                'X2', 'X1', TARGET_VALUE # Target: X2, Intervención: X1, Valor Objetivo: 1
            )
            results['X1_to_X2'] = {'cs': cs_x1_to_x2, 'interv': 'X1', 'target': 'X2', 'p_do_1': p_x1_1_x2, 'p_do_0': p_x1_0_x2}


            # 4. IMPRESIÓN DEL ANÁLISIS EN TERMINAL
            print("\n=======================================================")
            print("🔬 ANÁLISIS DE FUERZAS CAUSALES (Segun Creencias del Agente) 🧠")
            print(f"-------------------------------------------------------")
            
            for key, data in results.items():
                print(f"Relación: do({data['interv']}) -> {data['target']}")
                print(f"P({data['target']}=1 | do({data['interv']}=1)) = {data['p_do_1']:.4f}")
                print(f"P({data['target']}=1 | do({data['interv']}=0)) = {data['p_do_0']:.4f}")
                print(f"Fuerza Causal ({data['interv']} -> {data['target']}): {data['cs']:.4f}")
                print("-" * 40)
            
            print("=======================================================")
            
            # 5. VISUALIZACIÓN FINAL DEL DAG con todas las Fuerzas Causales
            cs_values_for_plot = {k: v['cs'] for k, v in results.items()}

            half_blind_agent.model.plot_dag(
                causal_strengths_dict=cs_values_for_plot,
                title='DAG con Todas las Fuerzas Causales Aprendidas'
            ) 
        
    except FileNotFoundError:
        print("\n!!! ERROR: Archivo de configuración no encontrado !!!")
        print(f"Asegúrate de que el archivo '{CONFIG_FILE_PATH}' exista.")
    except json.JSONDecodeError as e:
        print("\n!!! ERROR: Contenido JSON Inválido !!!")
        print(f"El archivo '{CONFIG_FILE_PATH}' no contiene JSON válido. Error: {e}")
    except Exception as e:
        print(f"\nOcurrió un error crítico durante la ejecución: {e}")