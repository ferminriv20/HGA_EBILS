"""

Este script integra los componentes del Algoritmo Genético Híbrido. Encapsula la configuración de hiperparámetros,
operadores genéticos, heurística de búsqueda local (ILS) y  las estrategias de control de flujo (parada y reinicio).
Facilita la implementación y prueba de configuraciones sin necesidad de modificar el núcleo del algoritmo

"""

import numpy as np
from GeneticAlgorithm_V2 import AlgoritmoGenetico
from Algorith_pMP import evaluar_poblacion, poblacion_inicial_combinaciones, selecciona_torneo, cruzamiento_intercambio , mutacion_simple_Swap, criterio_reinicio_adaptive, accion_reinicio, criterio_parada_estancamiento, _local_search_1_Swap_jit, Iterated_Local_Search  


def pmedian_hybrid( cost_matrix: np.ndarray, p: int, num_iteraciones: int , pop_size: int , seleccion : callable , cruzamiento: callable, mutacion:callable,   para_seleccion :dict , para_cruzamiento: dict, para_mutacion : dict,  prob_cruzamiento: float , prob_mutacion: float , max_estancamiento: int ,max_gen: int  ,min_gener: int ,  maximizar: bool = False) -> dict:
  
    # número total de instalaciones
    FACILITIES = cost_matrix.shape[0]
    # Población inicial usando combinaciones únicas
    POB = poblacion_inicial_combinaciones(pop_size, FACILITIES, p)
    
    # Selección: aceptar string o función y construir parámetros por defecto si es necesario
    para_seleccion = para_seleccion or {}
    if isinstance(seleccion, str):
        if seleccion  == 'selecciona_torneo':
            seleccion_fun = selecciona_torneo
            # poner valor por defecto si no se pasó
            para_seleccion = {**{"num_competidores": 8}, **para_seleccion} 
        else:
            raise ValueError("param 'seleccion' debe ser 'selecciona_torneo' o una función callable")
    elif callable(seleccion):
        seleccion_fun = seleccion
        if seleccion is selecciona_torneo and not para_seleccion:
            para_seleccion = {"num_competidores": 8}
    else:
        raise ValueError("param 'seleccion' inválido")

    # Cruzamiento: aceptar string o función
    para_cruzamiento = para_cruzamiento or {}
    if isinstance(cruzamiento, str):
        if cruzamiento == 'cruzamiento_intercambio':
            cruz_fun = cruzamiento_intercambio
            para_cruzamiento = {**{"facilities": FACILITIES}, **para_cruzamiento}
        else:
            raise ValueError("param 'cruzamiento' debe ser 'cruzamiento_intercambio' o una función callable")
    elif callable(cruzamiento):
        cruz_fun = cruzamiento
        para_cruzamiento = para_cruzamiento or {}
        if cruz_fun is cruzamiento_intercambio and "facilities" not in para_cruzamiento:
            para_cruzamiento["facilities"] = FACILITIES
    else:
        raise ValueError("param 'cruzamiento' inválido")

    # Mutación: aceptar string o función
    para_mutacion = para_mutacion or {}
    if isinstance(mutacion, str):
        if mutacion == "mutacion_simple_Swap":
            mut_fun = mutacion_simple_Swap
            para_mutacion = {**{"facilities": FACILITIES}, **para_mutacion}
        else:
            raise ValueError("param 'mutacion' debe ser  'mutacion_simple_Swap' o una función callable")
    elif callable(mutacion):
        mut_fun = mutacion
        if mutacion is mutacion_simple_Swap and not para_mutacion:
            para_mutacion = {"facilities": FACILITIES}
        else:
            para_mutacion = para_mutacion or {}
    else:
        raise ValueError("param 'mutacion' inválido")
    
    
    # Búsqueda local para el mejor(es) individuo(s)
    local_search_Elite = Iterated_Local_Search
    para_local_search_Elite = { "cost_matrix": cost_matrix}
   
    # Criterios de parada y reinicio
    criterio_parada = criterio_parada_estancamiento(max_gen, min_gener)
    criterio_reinicio = criterio_reinicio_adaptive(max_estancamiento)   
    accion_reinicio_ = accion_reinicio(FACILITIES, p)

    # Construir y ejecutar el algoritmo genético híbrido
    result = AlgoritmoGenetico(
        num_iteraciones= num_iteraciones,
        tam= pop_size,
        poblacion_inicial= POB,
        evaluacion= evaluar_poblacion,
        seleccion= seleccion_fun,
        cruzamiento= cruz_fun,
        mutacion= mut_fun,
        prob_mutacion= prob_mutacion,
        prob_cruzamiento= prob_cruzamiento,
        para_evaluacion= {"cost_matrix": cost_matrix},
        para_seleccion= para_seleccion,
        para_cruzamiento= para_cruzamiento,
        para_mutacion= para_mutacion,
        maximizar= maximizar,
        busqueda_elite= local_search_Elite,
        para_busqueda_elite= para_local_search_Elite,
        criterio_parada= criterio_parada,
        criterio_reinicio= criterio_reinicio,
        reinicio_poblacion= accion_reinicio_,
        ratio_reinicio= 0.9,
        paso_reinicio= 0.0,
    ).run()

    # Retornar resultados clave: tiempo, mejor_solución, mejor_ costo
    return result["tiempo"] , result["mejor"] ,result["resultado"] 


#prueba

if __name__ == "__main__":
    print("::::: Iniciando Algoritmo :::::")

    p = 100
    TAM =500
    MATRIX = np.load(r'datasets\pmed15.npy')
    
    print(pmedian_hybrid(
        cost_matrix=MATRIX,
        p=p,
        num_iteraciones=500,
        pop_size=TAM,
        seleccion= selecciona_torneo,
        cruzamiento= cruzamiento_intercambio,
        mutacion= mutacion_simple_Swap,
        prob_mutacion= 0.6,
        prob_cruzamiento= 0.95,
        para_seleccion= {},
        para_cruzamiento= {},
        para_mutacion= {},
        max_estancamiento= 30,
        max_gen= 100,
        min_gener= 20,
        maximizar= False)[0]  )  

 


