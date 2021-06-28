import numpy as np
import random
from vprtw_aco_figure import VrptwAcoFigure
from vrptw_base import VrptwGraph, PathMessage
from ant import Ant
from threading import Thread, Event
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import copy
import time
from multiprocessing import Process
from multiprocessing import Queue as MPQueue


class MultipleAntColonySystem:
    def __init__(self, graph: VrptwGraph, ants_num=10, beta=1, q0=0.1, whether_or_not_to_show_figure=True):
        super()
        # graphique Emplacement du nœud et informations sur la durée du service
        self.graph = graph
        # ants_num Nombre de fourmis
        self.ants_num = ants_num
        # vehicle_capacity indique la charge maximale de chaque véhicule
        self.max_load = graph.vehicle_capacity
        # beta L'importance d'éclairer l'information
        self.beta = beta
        # q0 indique la probabilité de sélectionner directement le prochain point avec la probabilité la plus élevée
        self.q0 = q0
        # best path
        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = None

        self.whether_or_not_to_show_figure = whether_or_not_to_show_figure

    @staticmethod
    def stochastic_accept(index_to_visit, transition_prob):
        """
       Roulette
        :param index_to_visit: a list of N index (list or tuple)
        :param transition_prob:
        :return: selected index
        """
        # calculate N and max fitness value
        N = len(index_to_visit)

        # normalize
        sum_tran_prob = np.sum(transition_prob)
        norm_transition_prob = transition_prob/sum_tran_prob

        # select: O(1)
        while True:
            # randomly select an individual with uniform probability
            ind = int(N * random.random())
            if random.random() <= norm_transition_prob[ind]:
                return index_to_visit[ind]

    @staticmethod
    def new_active_ant(ant: Ant, vehicle_num: int, local_search: bool, IN: np.numarray, q0: float, beta: int, stop_event: Event):
        """
        Explorez sur la carte en fonction du véhicule_num spécifié. Le numéro de véhicule utilisé ne peut pas être supérieur au nombre spécifié. Acs_time et acs_vehicle utiliseront cette méthode
        Pour acs_time, vous devez visiter tous les nœuds (le chemin est faisable), essayez de trouver un chemin avec une distance de déplacement plus courte
        Pour acs_vehicle, le num de véhicule utilisé sera un de moins que le nombre de véhicules utilisés par le meilleur chemin actuellement trouvé. Pour utiliser moins de véhicules, essayez de visiter les nœuds. Si tous les nœuds sont visités (le chemin est faisable), macs être averti
        :param ant:
        :param vehicle_num:
        :param local_search:
        :param IN:
        :param q0:
        :param beta:
        :param stop_event:
        :return:
        """
        # print('[new_active_ant]: start, start_index %d' % ant.travel_path[0])

       # Dans new_active_ant, vehicle_num véhicules peuvent être utilisés au maximum, c'est-à-dire que vehicle_num + 1 nœuds de dépôt peuvent être utilisés au maximum. Puisqu'un nœud de départ est utilisé, seuls les nœuds de dépôt de véhicules sont laissés
        unused_depot_count = vehicle_num

       # S'il y a des nœuds non visités, et vous pouvez retourner au dépôt
        while not ant.index_to_visit_empty() and unused_depot_count > 0:
            if stop_event.is_set():
                # print('[new_active_ant]: receive stop event')
                return

           # Calculer tous les prochains nœuds qui répondent à la charge et aux autres restrictions
            next_index_meet_constrains = ant.cal_next_index_meet_constrains()

            # S'il n'y a pas de nœud suivant qui respecte la limite, retournez au dépôt
            if len(next_index_meet_constrains) == 0:
                ant.move_to_next_index(0)
                unused_depot_count -= 1
                continue

            # Commencez à calculer le prochain nœud qui satisfait la restriction et sélectionnez la probabilité de chaque nœud
            length = len(next_index_meet_constrains)
            ready_time = np.zeros(length)
            due_time = np.zeros(length)

            for i in range(length):
                ready_time[i] = ant.graph.nodes[next_index_meet_constrains[i]].ready_time
                due_time[i] = ant.graph.nodes[next_index_meet_constrains[i]].due_time

            delivery_time = np.maximum(ant.vehicle_travel_time + ant.graph.node_dist_mat[ant.current_index][next_index_meet_constrains], ready_time)
            delta_time = delivery_time - ant.vehicle_travel_time
            distance = delta_time * (due_time - ant.vehicle_travel_time)

            distance = np.maximum(1.0, distance-IN[next_index_meet_constrains])
            closeness = 1/distance

            transition_prob = ant.graph.pheromone_mat[ant.current_index][next_index_meet_constrains] * \
                              np.power(closeness, beta)
            transition_prob = transition_prob / np.sum(transition_prob)

           # Sélectionnez directement le nœud avec la plus grande proximité selon la probabilité
            if np.random.rand() < q0:
                max_prob_index = np.argmax(transition_prob)
                next_index = next_index_meet_constrains[max_prob_index]
            else:
                # Utiliser l'algorithme de la roulette
                next_index = MultipleAntColonySystem.stochastic_accept(next_index_meet_constrains, transition_prob)

            # Mettre à jour la matrice de phéromones
            ant.graph.local_update_pheromone(ant.current_index, next_index)
            ant.move_to_next_index(next_index)

        # Si vous avez terminé tous les points, vous devez retourner au dépôt
        if ant.index_to_visit_empty():
            ant.graph.local_update_pheromone(ant.current_index, 0)
            ant.move_to_next_index(0)

        # Insérer des points non visités pour s'assurer que le chemin est faisable
        ant.insertion_procedure(stop_event)

        # ant.index_to_visit_empty()==Vrai signifie faisable
        if local_search is True and ant.index_to_visit_empty():
            ant.local_search_procedure(stop_event)

    @staticmethod
    def acs_time(new_graph: VrptwGraph, vehicle_num: int, ants_num: int, q0: float, beta: int,
                 global_path_queue: Queue, path_found_queue: Queue, stop_event: Event):
        """
        Pour acs_time, vous devez visiter tous les nœuds (le chemin est faisable), essayez de trouver un chemin avec une distance de déplacement plus courte
        :param new_graph:
        :param vehicle_num:
        :param ants_num:
        :param q0:
        :param beta:
        :param global_path_queue:
        :param path_found_queue:
        :param stop_event:
        :return:
        """

        # Un maximum de véhicules num_vehicule peut être utilisé, c'est-à-dire que le chemin avec la distance la plus courte peut être trouvé parmi les dépôts num_vehicule+1 au plus sur le chemin.
        # vehicle_num est défini pour être cohérent avec le best_path actuel
        print('[acs_time]: start, vehicle_num %d' % vehicle_num)
        # Initialiser la matrice de phéromones
        global_best_path = None
        global_best_distance = None
        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        while True:
            print('[acs_time]: new iteration')

            if stop_event.is_set():
                print('[acs_time]: receive stop event')
                return

            for k in range(ants_num):
                ant = Ant(new_graph, 0)
                thread = ants_pool.submit(MultipleAntColonySystem.new_active_ant, ant, vehicle_num, True,
                                          np.zeros(new_graph.node_num), q0, beta, stop_event)
                ants_thread.append(thread)
                ants.append(ant)

            # Vous pouvez utiliser la méthode result ici pour attendre la fin du thread
            for thread in ants_thread:
                thread.result()

            ant_best_travel_distance = None
            ant_best_path = None
            # Déterminer si le chemin trouvé par la fourmi est faisable et meilleur que le chemin global
            for ant in ants:

                if stop_event.is_set():
                    print('[acs_time]: receive stop event')
                    return

                # Obtenez le meilleur chemin actuel
                if not global_path_queue.empty():
                    info = global_path_queue.get()
                    while not global_path_queue.empty():
                        info = global_path_queue.get()
                    print('[acs_time]: receive global path info')
                    global_best_path, global_best_distance, global_used_vehicle_num = info.get_path_info()

                # Le chemin le plus court calculé par le chemin fourmi
                if ant.index_to_visit_empty() and (ant_best_travel_distance is None or ant.total_travel_distance < ant_best_travel_distance):
                    ant_best_travel_distance = ant.total_travel_distance
                    ant_best_path = ant.travel_path

            # Effectuez la mise à jour globale de la phéromone ici
            new_graph.global_update_pheromone(global_best_path, global_best_distance)

            # Envoyer le meilleur chemin actuel calculé aux macs
            if ant_best_travel_distance is not None and ant_best_travel_distance < global_best_distance:
                print('[acs_time]: ants\' local search found a improved feasible path, send path info to macs')
                path_found_queue.put(PathMessage(ant_best_path, ant_best_travel_distance))

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    @staticmethod
    def acs_vehicle(new_graph: VrptwGraph, vehicle_num: int, ants_num: int, q0: float, beta: int,
                    global_path_queue: Queue, path_found_queue: Queue, stop_event: Event):
        """
        Pour acs_vehicle, le num de véhicule utilisé sera un de moins que le nombre de véhicules utilisés par le meilleur chemin actuellement trouvé. Pour utiliser moins de véhicules, essayez de visiter les nœuds. Si tous les nœuds sont visités (le chemin est faisable), macs être averti
        :param new_graph:
        :param vehicle_num:
        :param ants_num:
        :param q0:
        :param beta:
        :param global_path_queue:
        :param path_found_queue:
        :param stop_event:
        :return:
        """
       # vehicle_num est défini sur un de moins que le best_path actuel
        print('[acs_vehicle]: start, vehicle_num %d' % vehicle_num)
        global_best_path = None
        global_best_distance = None

       # Utilisez l'algorithme le plus proche_neighbor_heuristic pour initialiser le chemin et la distance
        current_path, current_path_distance, _ = new_graph.nearest_neighbor_heuristic(max_vehicle_num=vehicle_num)

        # Découvrez les nœuds non visités dans le chemin actuel
        current_index_to_visit = list(range(new_graph.node_num))
        for ind in set(current_path):
            current_index_to_visit.remove(ind)

        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        IN = np.zeros(new_graph.node_num)
        while True:
            print('[acs_vehicle]: new iteration')

            if stop_event.is_set():
                print('[acs_vehicle]: receive stop event')
                return

            for k in range(ants_num):
                ant = Ant(new_graph, 0)
                thread = ants_pool.submit(MultipleAntColonySystem.new_active_ant, ant, vehicle_num, False, IN, q0,
                                          beta, stop_event)

                ants_thread.append(thread)
                ants.append(ant)

            # Vous pouvez utiliser la méthode result ici pour attendre la fin du thread
            for thread in ants_thread:
                thread.result()

            for ant in ants:

                if stop_event.is_set():
                    print('[acs_vehicle]: receive stop event')
                    return

                IN[ant.index_to_visit] = IN[ant.index_to_visit]+1

                # Comparez le chemin trouvé par les fourmis avec current_path, pouvez-vous utiliser vehicle_num pour accéder à plus de nœuds
                if len(ant.index_to_visit) < len(current_index_to_visit):
                    current_path = copy.deepcopy(ant.travel_path)
                    current_index_to_visit = copy.deepcopy(ant.index_to_visit)
                    current_path_distance = ant.total_travel_distance
                    # Et mettre IN à 0
                    IN = np.zeros(new_graph.node_num)

                    # Si ce chemin est faisable, il doit être envoyé à macs_vrptw
                    if ant.index_to_visit_empty():
                        print('[acs_vehicle]: found a feasible path, send path info to macs')
                        path_found_queue.put(PathMessage(ant.travel_path, ant.total_travel_distance))

           # Mettre à jour la phéromone dans new_graph, global
            new_graph.global_update_pheromone(current_path, current_path_distance)

            if not global_path_queue.empty():
                info = global_path_queue.get()
                while not global_path_queue.empty():
                    info = global_path_queue.get()
                print('[acs_vehicle]: receive global path info')
                global_best_path, global_best_distance, global_used_vehicle_num = info.get_path_info()

            new_graph.global_update_pheromone(global_best_path, global_best_distance)

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    def run_multiple_ant_colony_system(self, file_to_write_path=None):
        """
        Ouvrez un autre thread pour exécuter multiple_ant_colony_system, utilisez le thread principal pour dessiner
        :return:
        """
        path_queue_for_figure = MPQueue()
        multiple_ant_colony_system_thread = Process(target=self._multiple_ant_colony_system, args=(path_queue_for_figure, file_to_write_path, ))
        multiple_ant_colony_system_thread.start()

        # Voulez-vous montrer la figure
        if self.whether_or_not_to_show_figure:
            figure = VrptwAcoFigure(self.graph.nodes, path_queue_for_figure)
            figure.run()
        multiple_ant_colony_system_thread.join()

    def _multiple_ant_colony_system(self, path_queue_for_figure: MPQueue, file_to_write_path=None):
        """
        Appelez acs_time et acs_vehicle pour explorer le chemin
        :param path_queue_for_figure:
        :return:
        """
        if file_to_write_path is not None:
            file_to_write = open(file_to_write_path, 'w')
        else:
            file_to_write = None

        start_time_total = time.time()

        # Deux files d'attente sont nécessaires ici, time_what_to_do et vehicle_what_to_do, qui sont utilisées pour indiquer aux deux threads acs_time et acs_vehicle quel est le meilleur chemin actuel, ou pour les empêcher de calculer
        global_path_to_acs_time = Queue()
        global_path_to_acs_vehicle = Queue()

        # Une autre file d'attente, path_found_queue est le chemin réalisable calculé en recevant acs_time et acs_vehicle qui est meilleur que le meilleur chemin
        path_found_queue = Queue()

        # Initialiser en utilisant l'algorithme du voisin le plus proche
        self.best_path, self.best_path_distance, self.best_vehicle_num = self.graph.nearest_neighbor_heuristic()
        path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

        while True:
            print('[multiple_ant_colony_system]: new iteration')
            start_time_found_improved_solution = time.time()

            # Les informations du meilleur chemin actuel sont placées dans la file d'attente pour informer acs_time et acs_vehicle quel est le meilleur_path actuel
            global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_distance))
            global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_distance))

            stop_event = Event()

            # acs_vehicle, essayez d'explorer avec les véhicules self.best_vehicle_num-1 pour visiter plus de nœuds
            graph_for_acs_vehicle = self.graph.copy(self.graph.init_pheromone_val)
            acs_vehicle_thread = Thread(target=MultipleAntColonySystem.acs_vehicle,
                                        args=(graph_for_acs_vehicle, self.best_vehicle_num-1, self.ants_num, self.q0,
                                              self.beta, global_path_to_acs_vehicle, path_found_queue, stop_event))

           # acs_time Essayez d'explorer avec self.best_vehicle_num et trouvez un chemin plus court
            graph_for_acs_time = self.graph.copy(self.graph.init_pheromone_val)
            acs_time_thread = Thread(target=MultipleAntColonySystem.acs_time,
                                     args=(graph_for_acs_time, self.best_vehicle_num, self.ants_num, self.q0, self.beta,
                                           global_path_to_acs_time, path_found_queue, stop_event))

           # Démarrez acs_vehicle_thread et acs_time_thread, lorsqu'ils trouvent un chemin faisable et un meilleur chemin que le meilleur chemin, ils seront envoyés aux macs
            print('[macs]: start acs_vehicle and acs_time')
            acs_vehicle_thread.start()
            acs_time_thread.start()

            best_vehicle_num = self.best_vehicle_num

            while acs_vehicle_thread.is_alive() and acs_time_thread.is_alive():

                # Si aucun meilleur résultat n'est trouvé dans le délai spécifié, quittez le programme
                given_time = 10
                if time.time() - start_time_found_improved_solution > 60 * given_time:
                    stop_event.set()
                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    self.print_and_write_in_file(file_to_write, 'time is up: cannot find a better solution in given time(%d minutes)' % given_time)
                    self.print_and_write_in_file(file_to_write, 'it takes %0.3f second from multiple_ant_colony_system running' % (time.time()-start_time_total))
                    self.print_and_write_in_file(file_to_write, 'the best path have found is:')
                    self.print_and_write_in_file(file_to_write, self.best_path)
                    self.print_and_write_in_file(file_to_write, 'best path distance is %f, best vehicle_num is %d' % (self.best_path_distance, self.best_vehicle_num))
                    self.print_and_write_in_file(file_to_write, '*' * 50)

                    # Passer Aucun comme indicateur de fin
                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(PathMessage(None, None))

                    if file_to_write is not None:
                        file_to_write.flush()
                        file_to_write.close()
                    return

                if path_found_queue.empty():
                    continue

                path_info = path_found_queue.get()
                print('[macs]: receive found path info')
                found_path, found_path_distance, found_path_used_vehicle_num = path_info.get_path_info()
                while not path_found_queue.empty():
                    path, distance, vehicle_num = path_found_queue.get().get_path_info()

                    if distance < found_path_distance:
                        found_path, found_path_distance, found_path_used_vehicle_num = path, distance, vehicle_num

                    if vehicle_num < found_path_used_vehicle_num:
                        found_path, found_path_distance, found_path_used_vehicle_num = path, distance, vehicle_num

                # Si la distance du chemin trouvé (ce qui est faisable) est plus courte, mettez à jour les informations actuelles sur le meilleur chemin
                if found_path_distance < self.best_path_distance:

                    # Recherchez de meilleurs résultats, mettez à jour start_time
                    start_time_found_improved_solution = time.time()

                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    self.print_and_write_in_file(file_to_write, '[macs]: distance of found path (%f) better than best path\'s (%f)' % (found_path_distance, self.best_path_distance))
                    self.print_and_write_in_file(file_to_write, 'it takes %0.3f second from multiple_ant_colony_system running' % (time.time()-start_time_total))
                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    if file_to_write is not None:
                        file_to_write.flush()

                    self.best_path = found_path
                    self.best_vehicle_num = found_path_used_vehicle_num
                    self.best_path_distance = found_path_distance

                    # Si vous devez dessiner des graphiques, le meilleur chemin à trouver est envoyé au programme de dessin
                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

                    # Notifier les deux threads de acs_vehicle et acs_time, le best_path et best_path_distance actuellement trouvés
                    global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_distance))
                    global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_distance))

                # Si les chemins trouvés par ces deux threads utilisent moins de véhicules, arrêtez ces deux threads et démarrez l'itération suivante
                # Envoyer les informations d'arrêt à acs_time et acs_vehicle
                if found_path_used_vehicle_num < best_vehicle_num:

                    # Recherchez de meilleurs résultats, mettez à jour start_time
                    start_time_found_improved_solution = time.time()
                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    self.print_and_write_in_file(file_to_write, '[macs]: vehicle num of found path (%d) better than best path\'s (%d), found path distance is %f'
                          % (found_path_used_vehicle_num, best_vehicle_num, found_path_distance))
                    self.print_and_write_in_file(file_to_write, 'it takes %0.3f second multiple_ant_colony_system running' % (time.time() - start_time_total))
                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    if file_to_write is not None:
                        file_to_write.flush()

                    self.best_path = found_path
                    self.best_vehicle_num = found_path_used_vehicle_num
                    self.best_path_distance = found_path_distance

                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

                    # Arrêtez acs_time et acs_vehicle deux threads
                    print('[macs]: send stop info to acs_time and acs_vehicle')
                    # Notifier les deux threads de acs_vehicle et acs_time, le best_path et best_path_distance actuellement trouvés
                    stop_event.set()

    @staticmethod
    def print_and_write_in_file(file_to_write=None, message='default message'):
        if file_to_write is None:
            print(message)
        else:
            print(message)
            file_to_write.write(str(message)+'\n')
