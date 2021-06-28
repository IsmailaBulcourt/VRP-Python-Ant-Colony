import matplotlib.pyplot as plt
from multiprocessing import Queue as MPQueue


class VrptwAcoFigure:
    def __init__(self, nodes: list, path_queue: MPQueue):
        """
        Le calcul du dessin matplotlib doit être placé sur le fil principal, il est recommandé d'ouvrir un autre fil pour le travail de recherche du chemin.
        Lorsque le fil à la recherche d'un chemin trouve un nouveau chemin, placez le chemin dans path_queue, et le fil de dessin graphique dessinera automatiquement
        Le chemin stocké dans la file d'attente existe sous la forme de PathMessage (classe)
        Les nœuds stockés dans les nœuds existent sous la forme de Node (classe), qui utilisent principalement Node.x, Node.y pour obtenir les coordonnées du nœud
        :param nodes : les nœuds sont une liste de chaque nœud, y compris le dépôt
        :param path_queue : la file d'attente est utilisée pour stocker le chemin calculé par le thread de travail. Chaque élément de la file d'attente est un chemin et l'id de chaque nœud est stocké dans le chemin.
        """

        self.nodes = nodes
        self.figure = plt.figure(figsize=(10, 10))
        self.figure_ax = self.figure.add_subplot(1, 1, 1)
        self.path_queue = path_queue
        self._depot_color = 'k'
        self._customer_color = 'steelblue'
        self._line_color = 'darksalmon'

    def _draw_point(self):
        # Dessiner le dépôt
        self.figure_ax.scatter([self.nodes[0].x], [self.nodes[0].y], c=self._depot_color, label='depot', s=40)

        # Dégager le client
        self.figure_ax.scatter(list(node.x for node in self.nodes[1:]),
                               list(node.y for node in self.nodes[1:]), c=self._customer_color, label='customer', s=20)
        plt.pause(0.5)

    def run(self):
       # Dessinez d'abord chaque nœud
        self._draw_point()
        self.figure.show()

        # Lire le nouveau chemin dans la file d'attente et dessiner
        while True:
            if not self.path_queue.empty():
                # Prenez le dernier chemin de la file d'attente et supprimez les autres chemins
                info = self.path_queue.get()
                while not self.path_queue.empty():
                    info = self.path_queue.get()

                path, distance, used_vehicle_num = info.get_path_info()
                if path is None:
                    print('[draw figure]: exit')
                    break

               # Vous devez d'abord enregistrer la ligne à supprimer et vous ne pouvez pas la supprimer directement dans la première boucle.
                # Sinon, self.figure_ax.lines changera pendant la boucle, provoquant l'échec de la suppression de certaines lignes
                remove_obj = []
                for line in self.figure_ax.lines:
                    if line._label == 'line':
                        remove_obj.append(line)

                for line in remove_obj:
                    self.figure_ax.lines.remove(line)
                remove_obj.clear()

                # Redessiner la ligne
                self.figure_ax.set_title('travel distance: %0.2f, number of vehicles: %d ' % (distance, used_vehicle_num))
                self._draw_line(path)
            plt.pause(1)

    def _draw_line(self, path):
        # Dessiner le chemin en fonction de l'index dans le chemin
        for i in range(1, len(path)):
            x_list = [self.nodes[path[i - 1]].x, self.nodes[path[i]].x]
            y_list = [self.nodes[path[i - 1]].y, self.nodes[path[i]].y]
            self.figure_ax.plot(x_list, y_list, color=self._line_color, linewidth=1.5, label='line')
            plt.pause(0.2)
