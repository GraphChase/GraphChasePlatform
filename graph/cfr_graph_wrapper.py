import copy

class CfrmixGraph(object):
    def __init__(self, graph, initial_locations: list[list[int], list[int], list[int]], time_horizon):

        if hasattr(graph, 'row'):
            self.column = graph.column
            self.row = graph.row
        self.total_node_number = graph.num_nodes
        # self.adjacent = [[i] for i in range(1, self.total_node_number + 1)]
        # self.adjacent_not_i = [[] for _ in range(1, self.total_node_number + 1)]
        self.adjacent = [graph.adjlist[k] for k in graph.adjlist.keys()]
        self.adjacent_not_i = [[x for x in lst if x != (i+1)] 
                               for i, lst in enumerate(self.adjacent)]

        self.exit_node = initial_locations[-1]
        self.initial_locations = initial_locations
        self.stack_node = []
        self.time_horizon = time_horizon
        # self.build_graph()

    # Build the graph according to some rules
    def build_graph(self):
        count = 1
        for i in range(self.total_node_number):
            if i + self.column < self.total_node_number:
                self.adjacent[i].append(i + self.column + 1)
                self.adjacent[i + self.column].append(i + 1)

                self.adjacent_not_i[i].append(i + self.column + 1)
                self.adjacent_not_i[i + self.column].append(i + 1)

            if count != self.column:
                self.adjacent[i].append(i + 2)
                self.adjacent[i + 1].append(i + 1)

                self.adjacent_not_i[i].append(i + 2)
                self.adjacent_not_i[i + 1].append(i + 1)
                count += 1
            else:
                count = 1

    def get_neighbor_node(self, node_number):
        neighbor_node = self.adjacent_not_i[node_number - 1]
        return neighbor_node

    # Return the path from the specified node to the exit nodes and the length of path is less then a specified length
    # If flag is True, then the path has no circle otherwise the path may have circle
    def get_path(self, node_number_start, length, flag=True):
        path = []
        s1 = Stack()
        s2 = Stack()
        s1.push(node_number_start)
        s2.push(self.get_neighbor_node(node_number_start))
        while s1.items:
            while not s2.peek():
                s2.pop()
                s1.pop()
                if not s1.items:
                    break
            if not s1.items:
                break
            next_node = s2.peek()[0]
            s2.items[-1] = s2.peek()[1:]

            if flag:
                if next_node in s1.items:
                    continue

            s1.push(next_node)
            s2.push(self.get_neighbor_node(next_node))
            if next_node in self.exit_node and len(s1.items) <= length + 1:
                path.append(copy.deepcopy(s1.items))
                s2.pop()
                s1.pop()
            elif len(s1.items) >= length + 1:
                s2.pop()
                s1.pop()
        return path


'''define a Stack class'''


class Stack(object):
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]


# if __name__ == '__main__':
#     # for i in range(20):
#     #     for j in range(20):
#     #         print(j+1+i*20, end=" ")
#     #     print("\n")
#     time_horizon = 5
#     game_graph = graph(row=15, column=15, exit_node=[81, 85, 97, 99, 127, 129, 141, 145])
#     init_location = [113, (111, 115, 83, 143)]
#     print(len(game_graph.get_path(init_location[0], time_horizon, flag=False)))