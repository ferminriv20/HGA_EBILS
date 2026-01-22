"""

Este módulo contiene las implementaciones de un algoritmo genético híbrido aplicado al problema de la p-mediana (pMP).
Incluye la definición de la población inicial, las funciones de evaluación, selección, cruzamiento, mutación, así 
como los criterios de parada y reinicio, además de una heurística de búsqueda local iterativa.
Cada función está documentada con la descripción de sus argumentos y valores de retorno.
Asimismo, todas las funciones críticas han sido optimizadas mediante compilación JIT (Numba) para garantizar un alto 
rendimiento computacional.

"""

import numba
import numpy as  np 
from numba import jit, njit, prange
from GeneticAlgorithm_V2 import AlgoritmoGenetico
from poblacion_combinaciones import Combinaciones
from p_median_problem import PMedian


def poblacion_inicial_combinaciones(pop_size: int, facilities: int, p: int) -> np.ndarray:
    """
    Genera una población inicial de tamaño pop_size usando la clase Combinaciones,
    garantizando individuos únicos (combinaciones sin repetidos).

    Args:
        pop_size: Número de individuos a generar.
        facilities: Número total de instalaciones posibles (0..n-1).
        p: Número de instalaciones seleccionadas por individuo.
        
    Returns:
        np.ndarray: Matriz (pop_size, p) con individuos ordenados y sin duplicados.
        
    """
    combs = Combinaciones()
    combs.generar(n=facilities, m=p, tam=pop_size)   # llena combs.datos con 'pop_size' combinaciones

    # Convertimos el set de tuplas en un array (pop_size, p)
    poblacion = np.array(list(combs.datos), dtype=int)

    #si el orden del set no coincide con pop_size exacto
    if poblacion.shape[0] > pop_size:
        poblacion = poblacion[:pop_size, :]

    return poblacion

@njit(parallel=True, cache=True)
def evaluar_poblacion(poblacion: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """
    Evalúa el costo total de cada individuo en la población dada.
    
    Args:
        poblacion :  matriz que contiene  un conjunto de soluciones del pMP (cada fila es un individuo).
        cost_matrix : matriz de distancias clientes - instalaciones (n x n).
        
    Returns:
        np.ndarray : vector de costos totales para cada individuo.
        
    """
    pop_size = poblacion.shape[0]
    p = poblacion.shape[1]
    n = cost_matrix.shape[0]
    
    costos = np.empty(pop_size, dtype=np.float64)
    
    for i in prange(pop_size):
        # Evaluar individuo i
        total_cost = 0.0
        for c in range(n): # Para cada cliente
            # Encontrar distancia mínima a instalaciones del individuo
            min_dist = np.inf
            for k in range(p):
                fac = poblacion[i, k]
                d = cost_matrix[c, fac]
                if d < min_dist:
                    min_dist = d
            total_cost += min_dist
        costos[i] = total_cost
        
    return costos

@njit(cache=True, fastmath=True)
def evaluar_individuo_rapido(individuo : np.ndarray , cost_matrix: np.ndarray) -> float:
    """
    Evalúa el costo total de un individuo específico.
    
    Args:
        individuo: array 1D con índices de instalaciones seleccionadas
        cost_matrix: matriz de distancias clientes - instalaciones (n x n)
        
    Returns:
        float: costo total del individuo
        
    """
    n_clientes = cost_matrix.shape[0]
    costo_total = 0.0
    
    for cliente in range(n_clientes):
        dist_min = np.inf
        for instalacion in individuo:
            dist = cost_matrix[cliente, instalacion]
            if dist < dist_min:
                dist_min = dist
        costo_total += dist_min
    
    return costo_total

@njit(parallel=True, cache=True)
def selecciona_torneo(poblacion: np.ndarray, fitness : np.ndarray , num_elegidos: int, num_competidores, maximizar=False) -> np.ndarray:
    """
    Selecciona padres usando el método de torneo.
    
    Args:
        poblacion:  Matriz (pop_size, p) con la población actual.
        fitness:  Vector (pop_size,) con los valores de fitness de cada individuo.
        num_elegidos: Número de padres a seleccionar. (no se pasa este parametro por que es fijo y es equivalente al tamaño de la población)
        num_competidores: Número de individuos que compiten en cada torneo.
        maximizar: Indica si se busca maximizar o minimizar el fitness.
        
    Returns:
        np.ndarray: Matriz (num_elegidos, p) con los padres seleccionados.
            
    """
    n_pop = len(poblacion) 
    p = poblacion.shape[1]
    padres = np.empty((num_elegidos, p), dtype=poblacion.dtype)
    
    # Pre-calcular el signo para usar argmin siempre
    mult = -1.0 if maximizar else 1.0
    
    for i in prange(num_elegidos):
        # Selección aleatoria de competidores sin reemplazo
        # Numba no soporta choice(size=k, replace=False) eficientemente en parallel loops antiguos 
        # pero para k pequeño podemos hacerlo manualmente o usar choice con replace si n_pop >> k
        # Aquí usamos un enfoque simple válido en numba
        
        best_idx = -1
        best_fit = np.inf
        
        for _ in range(num_competidores):
            idx = np.random.randint(0, n_pop)
            fit = fitness[idx] * mult
            
            if fit < best_fit:
                best_fit = fit
                best_idx = idx
        
        padres[i] = poblacion[best_idx]
    return padres

def cruzamiento_intercambio(padre1: np.ndarray, padre2: np.ndarray, facilities : int) -> tuple[np.ndarray, np.ndarray]:
    """
    Cruza dos padres intercambiando segmentos de sus genes no comunes.
      IMPORTANTE : Si los padres son idénticos, se generan 2 hijos aleatorios.
      
    Args:
        padre1, padre2: Padres a cruzar (vectores 1D de longitud p).
        facilities: Número total de posibles instalaciones.
        
    Returns:
        hijo1, hijo2:  Hijos resultantes del cruzamiento. 
    
    """
    p = padre1.size # Tamaño de los padres
    # Usar máscaras booleanas para operaciones de conjuntos O(N)
    mask_p1 = np.zeros(facilities, dtype=np.bool_)
    mask_p2 = np.zeros(facilities, dtype=np.bool_)
    # Marcar presencia
    mask_p1[padre1] = True
    mask_p2[padre2] = True
    # Genes comunes: presentes en p1 AND presentes en p2
    # Tomamos de padre1 los que están marcados en p2
    v_fixed = padre1[mask_p2[padre1]]    # Exclusivos de p1: presentes en p1 AND NOT presentes en p2
    v_ex1 = padre1[~mask_p2[padre1]]
    # Exclusivos de p2: presentes en p2 AND NOT presentes en p1
    v_ex2 = padre2[~mask_p1[padre2]]
    q = v_ex1.size  # Número de genes intercambiables
    # Caso trivial: si no hay parte intercambiable (padres idénticos),
    # los hijos seran copias aleatorias de instalaciones
    if q == 0:
        hijo1 = np.random.choice(facilities, size=p, replace=False)
        hijo2 = np.random.choice(facilities, size=p, replace=False)
        # Como trabajamos con conjuntos, es buena idea ordenar los genes        
        return np.sort(hijo1),np.sort(hijo2)
    # Elegir punto de corte k entre 1 y q-1
    if q == 1:
        # Solo un gen en intercambio: intercambiarlo completo
        k = 1
    else:
        k = np.random.randint(1,q)  # valores entre 1 y q-1
    # Construir las partes intercambiadas
    # hijo1: comunes + ex1[0:k] + ex2[k:q]
    # hijo2: comunes + ex2[0:k] + ex1[k:q]
    parte1_h1 = v_ex1[:k]
    parte2_h1 = v_ex2[k:]
    parte1_h2 = v_ex2[:k]
    parte2_h2 = v_ex1[k:]

    hijo1 = np.concatenate((v_fixed, parte1_h1, parte2_h1))
    hijo2 = np.concatenate((v_fixed, parte1_h2, parte2_h2))
    # Ordenar genes para mantener consistencia
    
    return np.sort(hijo1), np.sort(hijo2)

@njit(cache=True)
def mutacion_simple_Swap(individuo: np.ndarray, facilities: int) -> np.ndarray:
    """
    Reemplaza UNA instalación del individuo por otra que no esté presente.
    
    Args:
        individuo: Vector 1D de longitud p con índices únicos (0..n-1).
        facilities: Número total de posibles instalaciones. 
        
    Returns:
        np.ndarray: Individuo mutado.
        
    """
    mutado = individuo.copy()
    p = mutado.size
    pos = np.random.randint(0, p)
    
    # Buscar aleatoriamente un candidato que no esté en el individuo
    while True:
        candidate = np.random.randint(0, facilities)
        
        # Verificar pertenencia manualmente (más rápido en numba para arrays pequeños/medianos)
        is_present = False
        for val in mutado:
            if val == candidate:
                is_present = True
                break
        
        if not is_present:
            mutado[pos] = candidate
            break
            
    # Mantener orden para consistencia
    mutado.sort()
    return mutado

def criterio_reinicio_adaptive(max_estancamiento: int) -> callable:
    """
    Se realiza  un reinicio  de la poblacion si el mejor individuo no ha mejorado en un 
    numero determinado de generaciones.
    
    Args:
        max_estancamiento: Número máximo de generaciones sin mejora.

    Returns:
        función criterio(generacion: int, fitness: np.ndarray) -> bool
         que devuelve True si se debe reiniciar, False en caso contrario.
         
    """
    # Estado mutable para rastrear el estancamiento
    estado = {
        'mejor_fit_historico': float('inf'),
        'contador_estancamiento': 0,
        'ultimo_reinicio': 0
    }

    def criterio(paso_ignorado: float, generacion:int, fitness: np.ndarray) -> bool:
        # Evitar chequeos en la misma generación del reinicio
        if generacion == estado['ultimo_reinicio']:
            return False

        # se actualiza el estado de estancamiento
        mejor_actual = np.min(fitness)
        
        # Si hay mejora real 
        if mejor_actual < estado['mejor_fit_historico'] - 1e-6:
            estado['mejor_fit_historico'] = mejor_actual
            estado['contador_estancamiento'] = 0
        else:
            estado['contador_estancamiento'] += 1
        # Criterio de Estancamiento 
        if estado['contador_estancamiento'] >= max_estancamiento:
            print(f"   [ALERTA] Estancamiento por {max_estancamiento} gens. REINICIO FORZADO.")
            # Reseteamos contadores para dar tiempo a la nueva población
            estado['contador_estancamiento'] = 0 
            estado['ultimo_reinicio'] = generacion
            return True
        
        return False

    return criterio

def accion_reinicio(facilities: int, p: int)-> callable:
    """
    Conserva al mejor individuo y reinicia el resto de la población con nuevos individuos aleatorios.
    
    Args:
        facilities: Número total de posibles instalaciones. 
        p: Número de instalaciones seleccionadas por individuo.
        
    Returns:
        función accion(poblacion: np.ndarray, fitness: np.ndarray, ratio_ignorado: float)-> None
         que modifica la población in-place.
        
    """
    #
    def accion(poblacion: np.ndarray, fitness: np.ndarray, ratio_ignorado = 0) -> None:
        #  Encontrar al mejor individuo
        # Asumimos minimización. Si fuera maximización usar np.argmax
        idx_mejor = np.argmin(fitness)
        # Copiamos al mejor individuo para que no se pierda al sobrescribir
        mejor = poblacion[idx_mejor].copy()
        fitness_mejor = fitness[idx_mejor] # Solo para el print
       
        #Generar nueva población (N-1 individuos)
        pop_size = len(poblacion)
        num_nuevos = pop_size - 1
        # Generar aleatorios
        genes_nuevos = poblacion_inicial_combinaciones(num_nuevos, facilities, p)
        # guardamos al mejor individuo en la posición 0
        poblacion[0] = mejor
        
        # Llenamos el resto con sangre nueva
        limit = min(len(genes_nuevos), num_nuevos)
        poblacion[1 : 1+limit] = genes_nuevos[:limit]
        
        print(f"   >>> [REINICIO RÁPIDO] Mejor Individuo (Fit: {fitness_mejor:.1f}) salvado. {limit} renovados.")

    return accion

def criterio_parada_estancamiento(max_gen: int, min_gener: int) -> callable:
    """
    Criterio de parada basado en estancamiento del mejor fitness.
    
    Args:
        max_gen: Número máximo de generaciones sin mejora.
        min_gener: Número mínimo de generaciones antes de considerar parada.
        
    Returns:
        función criterio(generacion: int, fitness: np.ndarray) -> bool
         que devuelve True si se debe parar, False en caso contrario.
    
    """
    estado = {'mejor_fit': float('inf'), 'contador': 0}

    def criterio(generacion : int, fitness : np.ndarray) -> bool:
        if generacion < min_gener: # No detener el algortimo antes de min_gener
            return False
            
        mejor_actual = np.min(fitness) # Asumiendo minimización
        
        # Si mejora (con pequeña tolerancia por errores de flotante)
        if mejor_actual < estado['mejor_fit'] - 1e-6:
            estado['mejor_fit'] = mejor_actual # hubo mejora
            estado['contador'] = 0
            # print(f"  >> Mejora en gen {generacion}: {mejor_actual}")
        else:
            estado['contador'] += 1 # No hubo mejora
            
        if estado['contador'] >= max_gen: #se veridica si se alcanzo el maximo de generaciones sin mejora
            print(f"STOP: Estancamiento detectado  gen {generacion}.  {max_gen} gen sin mejora.")
            return True
        return False

    return criterio

@njit(cache=True, fastmath=True)
def _local_search_1_Swap_jit(individuo : np.ndarray, cost_matrix: np.ndarray, max_iter: int) -> tuple[np.ndarray, float]:
    """
    Busqueda local 1-Swap best-improvement, toma un individuo y lo mejora iterativamente intercambiando
    una instalación por otra no presente en el individuo.
    
    Args:
        individuo: Vector 1D con índices de instalaciones seleccionadas.
        cost_matrix: Matriz de distancias clientes - instalaciones (n x n).
        max_iter: Número máximo de iteraciones sin mejora antes de detenerse.
        
    Returns:
        np.ndarray: Individuo mejorado.
        float: Costo del individuo mejorado.
    
    Importante: Versión optimizada con Numba y evaluación incremental (Delta Evaluation).
    Evita recalcular toda la matriz de costos y elimina la creación de objetos temporales.
    """
    curr_ind = individuo.copy()
    p = curr_ind.size
    n_clientes = cost_matrix.shape[0]
    n_facilities = cost_matrix.shape[1]
    
    # EVALUACIÓN INICIAL Y ESTRUCTURAS DE SOPORTE
    # Necesitamos saber para cada cliente:
    # - Su nearest facility actual (índice en curr_ind y distancia)
    # - (Opcional) su segunda nearest para updates rápidos, pero con N recalcular es barato con Numba
    nearest_idx_in_ind = np.empty(n_clientes, dtype=np.int32) # Índice k en curr_ind tal que curr_ind[k] es el más cercano
    nearest_dist = np.empty(n_clientes, dtype=np.float64)
    
    curr_cost = 0.0
    
    for c in range(n_clientes):
        min_d = np.inf
        min_k = -1
        for k in range(p):
            fac = curr_ind[k]
            d = cost_matrix[c, fac]
            if d < min_d:
                min_d = d
                min_k = k
        nearest_idx_in_ind[c] = min_k
        nearest_dist[c] = min_d
        curr_cost += min_d
        
    # BUCLE PRINCIPAL
    for _ in range(max_iter):
        best_gain = 0.0
        best_swap_i = -1 # índice en curr_ind a quitar
        best_swap_j = -1 # facilidad (0..n-1) a poner
        
        # Probar todos los posibles swaps
        # i: índice de la mediana en curr_ind a remover
        for i in range(p):
            # j: nueva facility  a insertar
            # Para hacerlo rápido, podemos probar solo un subconjunto o todas. Aquí probamos todas las que no están en la solución.
            for j in range(n_facilities):
                # Verificar si j ya está en curr_ind.
                # Como p suele ser pequeño, búsqueda lineal es aceptable en registro.
                # O podríamos usar un array de booleanos pre-computado.
                is_present = False
                for k in range(p):
                    if curr_ind[k] == j:
                        is_present = True
                        break
                if is_present:
                    continue
                
                # CALCULAR DELTA (Ganancia)
                # gain = current_cost - new_cost (si gain > 0, mejora)
                # delta = new_cost - current_cost (si delta < 0, mejora)
                delta = 0.0
                
                # Vectorización manual sobre clientes
                for c in range(n_clientes):
                    # Costo actual del cliente: nearest_dist[c]
                    current_d = nearest_dist[c]
                    new_fac_d = cost_matrix[c, j]
                    
                    if nearest_idx_in_ind[c] == i:
                        # Este cliente estaba servido por la facility que estamos QUITANDO .
                        # Su nueva distancia será min(su segunda mejor opción, new_fac_d)
                        # Buscar segunda mejor opción en las (p-1) restantes
                        second_best = np.inf
                        for k in range(p):
                            if k == i: continue
                            val = cost_matrix[c, curr_ind[k]]
                            if val < second_best:
                                second_best = val
                        
                        new_d = min(second_best, new_fac_d)
                        delta += (new_d - current_d)
                        
                    else:
                        # El cliente estaba servido por otra facility que NO se quita.
                        # Su distancia solo mejora si new_fac_d es mejor que lo que ya tenía.
                        if new_fac_d < current_d:
                            delta += (new_fac_d - current_d)
                        # Si no, delta += 0 (se queda igual)
                
                # Comparar con mejor encontrado
                # Queremos maximizar gain = -delta, o minimizar delta
                gain = -delta
                if gain > best_gain + 1e-6:
                    best_gain = gain
                    best_swap_i = i
                    best_swap_j = j
        
        # SI HUBO MEJORA, APLICAR Y ACTUALIZAR ESTRUCTURAS
        if best_swap_i != -1:
            # Actualizar individuo
            curr_ind[best_swap_i] = best_swap_j
            # Nota: No ordenamos inmediatamente curr_ind para no romper la correspondencia de índices
            # con nearest_idx_in_ind, PERO el algoritmo asume que nearest_idx_in_ind apunta a posiciones.
            # Al cambiar curr_ind[best_swap_i], la posición best_swap_i ahora tiene la nueva facility.
            # Actualizar costo actual
            curr_cost -= best_gain
            
            # Actualizar nearest structures para todos los clientes
            # Esto es más rápido que recalcular desde cero
            for c in range(n_clientes):
                current_d = nearest_dist[c]
                new_fac_d = cost_matrix[c, best_swap_j]
                
                if nearest_idx_in_ind[c] == best_swap_i:
                    # Perdió su nearest anterior (que estaba en best_swap_i)
                    # Ahora en best_swap_i hay una nueva facility.
                    # Debemos re-escanear TODO el vector curr_ind para encontrar el nuevo mínimo.
                    # (Incluyendo la nueva facility en best_swap_i)
                    min_d = np.inf
                    min_k = -1
                    for k in range(p):
                        fac = curr_ind[k]
                        d = cost_matrix[c, fac]
                        if d < min_d:
                            min_d = d
                            min_k = k
                    nearest_idx_in_ind[c] = min_k
                    nearest_dist[c] = min_d
                
                else:
                    # Su nearest anterior sigue ahí (en una posición != best_swap_i).
                    # Solo chequear si la nueva facility (en best_swap_i) es mejor.
                    if new_fac_d < current_d:
                        nearest_dist[c] = new_fac_d
                        nearest_idx_in_ind[c] = best_swap_i
                        
        else:
            # No hubo mejora en todo el vecindario
            break
            
    # Al final, ordenamos el individuo para retornarlo en formato canónico
    curr_ind.sort()
    
    # Recalculamos costo final exacto por si acaso hubo drift numérico (opcional, pero seguro)
    # final_cost = evaluar_individuo_rapido(curr_ind, cost_matrix)
    # O confiamos en curr_cost acumulado
    
    return curr_ind, curr_cost

def Iterated_Local_Search(individuo: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """
    Busqueda Local Iterada (ILS) para el pMP. Combina búsqueda local exhaustiva
    con perturbaciones para escapar de óptimos locales.
    
    Args:
        individuo: Vector 1D con índices de instalaciones seleccionadas.
        cost_matrix: Matriz de distancias clientes - instalaciones (n x n).
    
    Returns:
        np.ndarray: Individuo mejorado tras aplicar ILS.
    
    """
    mejor_ind = individuo.copy()
    facilities = cost_matrix.shape[1]
    # Evaluación inicial
    costo_original = evaluar_individuo_rapido(mejor_ind, cost_matrix)
    # Pulido Exhaustivo Inicial (con límite)
    ind_pulido, costo_pulido = _local_search_1_Swap_jit(mejor_ind, cost_matrix, max_iter=10)
    
    if costo_pulido < costo_original - 1e-6:
        # Si hubo mejora en la fase inicial, retornamos el individuo pulido
        return ind_pulido
    
    # Perturbacion (escapar de óptimo local)
    p = individuo.size
 
    # Seleccionamos k aleatorio entre 2 y 5 (o menos si p es pequeño)
    max_k = min(5, p)
    if max_k < 2: 
        return ind_pulido # No se puede perturbar

    k_destruir = np.random.randint(2, max_k + 1) # randint es exclusivo en el límite superior
    vecino_kick = ind_pulido.copy()
    
    idxs_fuera = np.random.choice(p, k_destruir, replace=False)
    disponibles = np.setdiff1d(np.arange(facilities), vecino_kick, assume_unique=True)
    
    if disponibles.size < k_destruir:
        return ind_pulido # No hay suficientes instalaciones para cambiar
    
    nuevos = np.random.choice(disponibles, k_destruir, replace=False)
    vecino_kick[idxs_fuera] = nuevos
    vecino_kick.sort()
    
    # Reparar con local search (menos iteraciones que la fase inicial)
    vecino_reparado, costo_reparado = _local_search_1_Swap_jit( vecino_kick, cost_matrix,  max_iter=5)
   
   # Solo aceptamos el salto si aterrizamos en una solución mejor que donde estábamos estancados.
    if costo_reparado < costo_pulido - 1e-6:
        return vecino_reparado
    
    # Si el salto no sirvió, nos quedamos con el óptimo local original
    return ind_pulido


if __name__ == '__main__':
    
    #* PARÁMETROS DEL PROBLEMA *#
    
    PROBLEM_ID = 15
    pmed = PMedian(PROBLEM_ID)
    
    MATRIX = np.load(f"datasets/pmed{PROBLEM_ID}.npy")
    CLIENTS = len(MATRIX)
    FACILITIES = len(MATRIX)
    P = pmed.p # Número de medianas a seleccionar 
    
    TAM = 500 # Tamaño de la población

    criterio_parada_config= criterio_parada_estancamiento(max_gen=100, min_gener=0)
    criterio_reinicio_config = criterio_reinicio_adaptive(max_estancamiento=30)
    accion_reinicio_config = accion_reinicio(facilities= FACILITIES, p=P) 
    

    POB = poblacion_inicial_combinaciones(TAM, FACILITIES, P)
    
    print(
        'El tiempo de ejecución fue: ',
        AlgoritmoGenetico(
            num_iteraciones= 500,
            tam= TAM,
            poblacion_inicial= POB,
            evaluacion= evaluar_poblacion,
            seleccion= selecciona_torneo,
            cruzamiento = cruzamiento_intercambio,
            mutacion=  mutacion_simple_Swap,
            busqueda_elite= Iterated_Local_Search,
            prob_mutacion= 0.6,
            prob_cruzamiento= 0.95,
            para_evaluacion= {'cost_matrix': MATRIX},
            para_seleccion= {'num_competidores': 12}, #torneo
            # para_seleccion= {}, #ruleta
            para_cruzamiento= {'facilities': FACILITIES },
            para_mutacion= {'facilities': FACILITIES }, #mutación simple              
            # para_mutacion= {'cost_matrix': MATRIX, 'num_vecinos_cercanos': 20 }, #mutación geográfica
            para_busqueda_elite= {'cost_matrix': MATRIX }, #mutación por búsqueda local
            criterio_parada= criterio_parada_config,
            criterio_reinicio= criterio_reinicio_config,
            reinicio_poblacion= accion_reinicio_config,
            ratio_reinicio= 0.0,
            paso_reinicio= 0.0,
            maximizar= False
            ).run()['tiempo'], 'minutos. ')


