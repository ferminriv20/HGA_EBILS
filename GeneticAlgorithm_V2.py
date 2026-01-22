import numpy as np
import time


class AlgoritmoGenetico:
    def __init__(
            self, 
            num_iteraciones: int, 
            tam: int, 
            poblacion_inicial: np.ndarray,
            evaluacion: callable,
            seleccion: callable,
            cruzamiento: callable,
            mutacion: callable,
            prob_mutacion: float, 
            prob_cruzamiento: float, 
            para_evaluacion= dict(),
            para_seleccion= dict(),
            para_cruzamiento= dict(),
            para_mutacion= dict(),
            cruzar_pares: bool = False, 
            criterio_parada: callable= lambda x,y: False, # (generacion, fitness)
            criterio_reinicio: callable = lambda x,y,z: False, # (paso_reinicio, generacion, fitness)
            reinicio_poblacion: callable = False, # (poblacion, ratio)
            ratio_reinicio: float = 0.95,
            paso_reinicio: float = 0.2, # Cada tanto porcentaje del número de iteracinoes se permite hacer reinicio
            mejor: np.ndarray = None,
            busqueda_elite: callable = None,
            para_busqueda_elite: dict = None,
            maximizar: bool = False
            ):
        
        self.num_iteraciones = num_iteraciones
        self.tam = tam
        self.poblacion = poblacion_inicial
        self.criterio_parada = criterio_parada
        self.metodo_evaluacion = evaluacion
        self.metodo_seleccion = seleccion
        self.metodo_cruzamiento = cruzamiento
        self.metodo_mutacion = mutacion
        self.prob_mutacion = prob_mutacion
        self.prob_cruzamiento = prob_cruzamiento
        self.para_evaluacion = para_evaluacion
        self.para_seleccion = para_seleccion
        self.para_cruzamiento = para_cruzamiento
        self.para_mutacion = para_mutacion
        self.cruzar_pares = cruzar_pares
        self.criterio_reinicio = criterio_reinicio
        self.reinicio_poblacion = reinicio_poblacion
        self.ratio_reinicio = ratio_reinicio
        self.paso_reinicio = paso_reinicio
        self.mejor = mejor
        self.busqueda_elite = busqueda_elite
        self.para_busqueda_elite = para_busqueda_elite
        self.maximizar = maximizar
        

    def evaluacion(self, pob):
        return self.metodo_evaluacion(pob, **self.para_evaluacion)

    def seleccion(self, pob, fitness):
        return self.metodo_seleccion(pob, fitness, self.tam, maximizar= self.maximizar, **self.para_seleccion)

    def mutacion(self, individuo):
        return self.metodo_mutacion(individuo, **self.para_mutacion)

    def cruzamiento(self, padre1, padre2):
        return self.metodo_cruzamiento(padre1, padre2, **self.para_cruzamiento)
    
    def cruzamiento_total(self, seleccionados):
        return self.metodo_cruzamiento(seleccionados, **self.para_cruzamiento)

    def busqueda_elite(self, individuo):
        return self.metodo_busqueda_elite(individuo, **self.para_busqueda_elite)

    def run(self):
        start = time.time()
        print('Iniciando Algoritmo... \n\n')
        # Evaluación Inicial
        fitness = self.evaluacion(self.poblacion)
        # Inicio de Iteraciones 
        for generacion in range(self.num_iteraciones):
            # Busqueda Local Élite
            if  (generacion + 1) % 10== 0:
                # Identificar el mejor valor actual
                best_val = np.amax(fitness) if self.maximizar else np.amin(fitness)
                # Encontrar TODOS los índices que tienen ese fitness
                # Usar tolerancia para float
                if self.maximizar:
                    indices_elite = np.where(fitness >= best_val*0.9999)[0]
                else:
                    indices_elite = np.where(fitness <= best_val*1.00001)[0]
                
                if len(indices_elite) > 2:
                    # Si hay muchos elististas, seleccionar muestra
                    indices_elite = np.random.choice(indices_elite, 2 if (generacion+1) % 20 == 0 else 1, replace=False)
                
                print(f"   [Iter {generacion}] Aplicando Busqueda Élite a {len(indices_elite)} individuos...")
                
              
                for idx in indices_elite:
                    individuo_original = self.poblacion[idx].copy()
                    costo_original = fitness[idx]
                    # Aplicar heurística
                    ind_mejorado = self.busqueda_elite(individuo_original, **self.para_busqueda_elite)
                    # Evaluar solo el individuo mejorado
                    # Nota: evaluacion espera array 2D si vectorizado, pero suele manejar 1D si es robusto.
                    # Aseguramos formato (1, p)
                    costo_nuevo_arr = self.evaluacion(np.array([ind_mejorado]))
                    costo_nuevo = costo_nuevo_arr[0]
    
                    # Solo actualizamos si mejoró
                    if costo_nuevo < costo_original: # Asumiendo minimización para la decisión local
                        self.poblacion[idx] = ind_mejorado
                        fitness[idx] = costo_nuevo # Actualización incremental
                        print(f"     -> Individuo {idx} mejorado de {costo_original} a {costo_nuevo}")
                    else:
                        print(f"     -> Individuo {idx} no mejoró (costo {costo_original})")
            # Criterio de Reinicio 
            if self.criterio_reinicio(self.paso_reinicio, generacion, fitness):
                self.reinicio_poblacion(self.poblacion, fitness, self.ratio_reinicio)
                # Tras reinicio masivo, evaluación completa obligatoria
                fitness = self.evaluacion(self.poblacion)
                print(f' ### Reinicio de Población en la generación {generacion}\n')

            # Guardar el mejor actual
            idx_mejor_global = np.argmax(fitness) if self.maximizar else np.argmin(fitness)
            mejor = self.poblacion[idx_mejor_global].copy()
            mejor_score = fitness[idx_mejor_global]

            # Criterio de Parada
            if self.criterio_parada(generacion, fitness):
                break

            # Selección 
            seleccionados = self.seleccion(self.poblacion, fitness)

            # Cruzamiento 
            descendencia = []
            if self.cruzar_pares:
                descendencia = self.cruzamiento_total(seleccionados)
            else:
                num_cruces = self.tam // 2 # Aproximación para loop
                # Generamos descendencia hasta cubrir tamaño aprox o fijo
                # El original iteraba tam+1 veces? "for _ in range(self.tam + 1):"
                # Intentaremos mantener lógica original pero eficiente
                # Pre-generar pares random es más rápido
                rand_vals = np.random.rand(self.tam + 1)
                indices_padres = np.random.choice(self.tam, size=(self.tam + 1, 2), replace=True)
                
                lista_hijos = []
                # Este bucle sigue siendo python, podría optimizarse si cruzamiento fuese vectorizado total
                for k in range(self.tam + 1):
                    if rand_vals[k] > self.prob_cruzamiento:
                        continue
                    p1, p2 = indices_padres[k]
                    # cruzamiento devuelve tuple (h1, h2) o array
                    hijos = self.cruzamiento(seleccionados[p1], seleccionados[p2])
                    
                    # Aplanar
                    if isinstance(hijos, tuple):
                        lista_hijos.extend(hijos)
                    elif isinstance(hijos, np.ndarray):
                        if hijos.ndim > 1:
                            for row in hijos: lista_hijos.append(row)
                        else:
                            lista_hijos.append(hijos)
                    else:
                        lista_hijos.extend(hijos)
                
                descendencia = lista_hijos

            # Mutación
            # descendencia es lista de arrays
            for i in range(len(descendencia)):
                if np.random.rand() < self.prob_mutacion:
                    descendencia[i] = self.mutacion(descendencia[i])
        
            # Reemplazo y actualización de Fitness
            if len(descendencia) > 0:
                # Convertir a matriz para evaluación vectorizada (muy importante)
                desc_matrix = np.array(descendencia)
                # Evaluar descendencia en lote
                desc_fitness = self.evaluacion(desc_matrix)
                # Reemplazo aleatorio
                # Generar indices a reemplazar
                indices_reemplazo = np.random.randint(0, self.tam, size=len(descendencia))
                # Actualizar población y fitness de un golpe (si fuera posible)
                # O iterar. Numpy permite indexing avanzado.
                self.poblacion[indices_reemplazo] = desc_matrix
                fitness[indices_reemplazo] = desc_fitness
      
            # El mejor individuo de la generación ANTERIOR (guardado en 'mejor')
            # debe sobrevivir. Lo colocamos en la posición 0 (convenio común) 
            # o reemplazamos al peor. El código original hacía: self.poblacion[0] = mejor
            self.poblacion[0] = mejor
            fitness[0] = mejor_score

            # --- Reporte ---
            if (generacion+1) % 10 == 0: 
                print(f' Reporte Hasta Iteración #{generacion+1} ...')
                # Ya tenemos fitness actualizado
                self.reporte_de_estado(fitness, mejor= self.poblacion[np.argmax(fitness)] if self.maximizar else self.poblacion[np.argmin(fitness)])
            
        # Fin del bucle
        idx_final = np.argmax(fitness) if self.maximizar else np.argmin(fitness)
        mejor_individuo = self.poblacion[idx_final]
        resultado = fitness[idx_final]

        print('\n', f'Se realizaron un total de {generacion+1} iteraciones y se encontró que el individuo más apto fue:')
        print(f'  -> Individuo: \n{mejor_individuo}, \ncon una aptitud de {resultado}')
        print(' FIN EJECUCIÓN '.center(100, ':'))
        end = time.time()
        return {'pob': self.poblacion, 'aptitudes': fitness, 'mejor': mejor_individuo, 'resultado': resultado, 'tiempo': round((end-start)/60, 2)}
      
    # Método para reportar el estado del algoritmo:
    def reporte_de_estado(self, fitness, mejor):
        print(f'Mejor individuo: {mejor}\n')
        print(f'Aptitud: {np.amax(fitness) if self.maximizar else np.amin(fitness)}\n')
    #   print(f'Tamaño de la población: {len(fitness)}')
        print('-'*80)
        return None


    def __str__(self):
        return self.poblacion
    # Se crean los métodos de acceso a los atributos de la clase:
    @property
    def num_iteraciones(self):
        return self._num_iteraciones
    
    @num_iteraciones.setter
    def num_iteraciones(self, num_iteraciones):
        self._num_iteraciones = num_iteraciones

    @property
    def tam(self):
        return self._tam
    
    @tam.setter
    def tam(self, tam):
        self._tam = tam

    @property
    def poblacion(self):
        return self._poblacion
    
    @poblacion.setter
    def poblacion(self, poblacion):
        self._poblacion = poblacion

    @property
    def criterio_parada(self):
        return self._criterio_parada
    
    @criterio_parada.setter
    def criterio_parada(self, criterio_parada):
        self._criterio_parada = criterio_parada

    @property
    def metodo_evaluacion(self):
        return self._metodo_evaluacion
    
    @metodo_evaluacion.setter
    def metodo_evaluacion(self, metodo_evaluacion):
        self._metodo_evaluacion = metodo_evaluacion

    @property
    def metodo_seleccion(self):
        return self._metodo_seleccion
    
    @metodo_seleccion.setter
    def metodo_seleccion(self, metodo_seleccion):
        self._metodo_seleccion = metodo_seleccion

    @property
    def metodo_cruzamiento(self):
        return self._metodo_cruzamiento
    
    @metodo_cruzamiento.setter
    def metodo_cruzamiento(self, metodo_cruzamiento):
        self._metodo_cruzamiento = metodo_cruzamiento

    @property
    def metodo_mutacion(self):
        return self._metodo_mutacion
    
    @metodo_mutacion.setter
    def metodo_mutacion(self, metodo_mutacion):
        self._metodo_mutacion = metodo_mutacion

    @property
    def prob_mutacion(self):
        return self._prob_mutacion
    
    @prob_mutacion.setter
    def prob_mutacion(self, prob_mutacion):
        self._prob_mutacion = prob_mutacion

    @property
    def prob_cruzamiento(self):
        return self._prob_cruzamiento
    
    @prob_cruzamiento.setter
    def prob_cruzamiento(self, prob_cruzamiento):
        self._prob_cruzamiento = prob_cruzamiento

    @property
    def para_evaluacion(self):
        return self._para_evaluacion
    
    @para_evaluacion.setter
    def para_evaluacion(self, para_evaluacion):
        self._para_evaluacion = para_evaluacion

    @property
    def para_seleccion(self):
        return self._para_seleccion
    
    @para_seleccion.setter
    def para_seleccion(self, para_seleccion):
        self._para_seleccion = para_seleccion

    @property
    def para_cruzamiento(self):
        return self._para_cruzamiento
    
    @para_cruzamiento.setter
    def para_cruzamiento(self, para_cruzamiento):
        self._para_cruzamiento = para_cruzamiento

    @property
    def para_mutacion(self):
        return self._para_mutacion
    
    @para_mutacion.setter
    def para_mutacion(self, para_mutacion):
        self._para_mutacion = para_mutacion

    @property
    def cruzar_pares(self):
        return self._cruzar_pares
    
    @cruzar_pares.setter
    def cruzar_pares(self, cruzar_pares):
        self._cruzar_pares = cruzar_pares

    @property
    def criterio_reinicio(self):
        return self._criterio_reinicio
    
    @criterio_reinicio.setter
    def criterio_reinicio(self, criterio_reinicio):
        self._criterio_reinicio = criterio_reinicio
    
    @property
    def reinicio_poblacion(self):
        return self._reinicio_poblacion
    
    @reinicio_poblacion.setter
    def reinicio_poblacion(self, reinicio_poblacion):
        self._reinicio_poblacion = reinicio_poblacion

    @property
    def ratio_reinicio(self):
        return self._ratio_reinicio
    
    @ratio_reinicio.setter
    def ratio_reinicio(self, ratio_reinicio):
        self._ratio_reinicio = ratio_reinicio

    @property
    def paso_reinicio(self):
        return self._paso_reinicio
    
    @paso_reinicio.setter
    def paso_reinicio(self, paso_reinicio):
        self._paso_reinicio = paso_reinicio

    @property
    def mejor(self):
        return self._mejor
    
    @mejor.setter
    def mejor(self, mejor):
        self._mejor = mejor

    @property
    def maximizar(self):
        return self._maximizar
    
    @maximizar.setter
    def maximizar(self, maximizar):
        self._maximizar = maximizar
        
    