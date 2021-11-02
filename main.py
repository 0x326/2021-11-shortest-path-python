from dataclasses import dataclass
from math import inf
from typing import *

VertexId = str


class Edge(NamedTuple):
    from_vertex_id: VertexId
    to_vertex_id: VertexId
    cost: int


@dataclass
class Vertex:
    id: VertexId
    out_edges: set[Edge]


Cost = Union[int, float]


T = TypeVar('T')


@dataclass
class LinkedListNode:
    previous_node: Optional['LinkedListNode']
    next_node: Optional['LinkedListNode']
    value: T


class SortedSet:
    """
    Set implemented using a doubly linked list.
    Set order is guaranteed to be sorted.

    Inserts:
        - Best case (left side):   O(1)
        - Average case:            O(n)
        - Worst case (right side): O(n)

    Peak left:
        - O(1)

    Deletions:
        - Best case (left side):     O(1)  <-- This is the deletion we are using exclusively in this project
        - Average case (in general): O(n)
        - Worst case (right side):   O(n)
    """

    def __init__(self, values: Iterable[T] = ()):
        self.length = 0
        self.linked_list_first: Optional[LinkedListNode] = None
        self.linked_list_last: Optional[LinkedListNode] = None

        for value in values:
            self.add(value)

    def __repr__(self):
        return f'{self.__class__.__name__}({{{", ".join(repr(value) for value in self)}}})'

    def __iter__(self):
        current_node = self.linked_list_first
        while current_node is not None:
            yield current_node.value
            current_node = current_node.next_node

    def __reversed__(self):
        current_node = self.linked_list_last
        while current_node is not None:
            yield current_node.value
            current_node = current_node.previous_node

    def __len__(self):
        return self.length

    def peak_left(self) -> T:
        if len(self) == 0:
            raise KeyError(f'{self.__class__.__name__} is empty')
        value = self.linked_list_first.value
        return value

    def add(self, value: T):
        # Base case: Length == 0
        if self.linked_list_first is None:
            new_node = LinkedListNode(previous_node=None, next_node=None, value=value)
            self.linked_list_first = new_node
            self.linked_list_last = new_node
            self.length += 1
            return

        # Base case: Insertion point is position 0
        if value < self.linked_list_first.value:
            new_node = LinkedListNode(previous_node=None, next_node=self.linked_list_first, value=value)
            self.linked_list_first.previous_node = new_node
            self.linked_list_first = new_node
            self.length += 1
            return

        # Other cases: Insertion point is greater than 0
        # Find insertion point
        current_node = self.linked_list_first
        while not (current_node.value < value):
            if current_node.value == value:
                return  # value is already in this set
            if current_node.next_node is None:
                break  # We have reached the end
            current_node = current_node.next_node

        # Insert here
        # [current_node]  <--  .previous_node
        #    .next_node   -->  [next_node]
        # ------------------------------------------------------
        # [current_node]  <--  .previous_node
        #    .next_node   -->  [new_node]   <--  .previous_node
        #                      .next_node   -->  [next_node]
        next_node = current_node.next_node
        new_node = LinkedListNode(previous_node=current_node, next_node=next_node, value=value)
        current_node.next_node = new_node
        if next_node is not None:
            next_node.previous_node = new_node
        else:
            self.linked_list_last = new_node
        self.length += 1

    def remove(self, value: T):
        # Base case: Length == 0
        if self.linked_list_first is None:
            return

        # Base case: Length == 1
        if self.linked_list_first.next_node is None:
            if self.linked_list_first.value == value:
                self.linked_list_first = None
                self.linked_list_last = None
                self.length = 0
                return

        # Base case: Removal point is position 0 (Length >= 2)
        if value == self.linked_list_first.value:
            next_node = self.linked_list_first.next_node
            next_node.previous_node = None
            self.linked_list_first = next_node
            self.length -= 1
            return

        # Other cases: Removal point is greater than 0 (Length >= 2)
        # Find removal point
        current_node = self.linked_list_first
        while current_node.value != value:
            if current_node.value > value:
                return  # We would have past it already if it existed
            if current_node.next_node is None:
                return  # We have reached the end. It doesn't exit
            current_node = current_node.next_node

        # Remove current_node
        # [prev_node]  <--  .previous_node
        # .next_node   -->  [current_node]   <--  .previous_node
        #                       .next_node   -->  [next_node]
        # ------------------------------------------------------
        # [prev_node]  <--  .previous_node
        # .next_node   -->  [next_node]
        previous_node = current_node.previous_node
        next_node = current_node.next_node

        previous_node.next_node = next_node
        if next_node is not None:
            next_node.previous_node = previous_node
        else:
            self.linked_list_last = previous_node
        self.length -= 1


class SortedIndexDict:
    """
    A dictionary with a sorted index.
    Used to efficiently keep track of vertex costs in Dijkstra's algorithm.

    Insert/Update:
        1. Insert into dictionary: O(1)
        2. Insert into reverse dictionary: O(1) + O(underlying set implementation)
        3. Insert into SortedSet:  O(n), where n is the number of unique keys

    Deletion:
        1. Deletion from dictionary: O(1)
        2. Deletion from reverse dictionary: O(underlying set implementation)
        3. Deletion from SortedSet: O(1), since we delete the smallest element

    Get const: O(1)
    """
    def __init__(self, values: Iterable[Tuple[VertexId, Cost]] = ()):
        self.cost: dict[VertexId, Cost] = dict()
        self.index: dict[Cost, set[VertexId]] = dict()
        self.keys: SortedSet = SortedSet()

        for vertex_id, cost in values:
            self.update(vertex_id=vertex_id, cost=cost)

    def __repr__(self):
        return f'{self.__class__.__name__}(({", ".join(f"({key}, {value})" for key, value in self.cost.items())}))'

    def __len__(self):
        return len(self.cost)

    def get_cost(self, vertex_id: VertexId) -> Cost:
        return self.cost[vertex_id]

    def pop_lowest(self) -> Tuple[VertexId, Cost]:
        cost = self.keys.peak_left()
        vertex_id = self.index[cost].pop()
        del self.cost[vertex_id]
        if len(self.index[cost]) == 0:
            del self.index[cost]
            self.keys.remove(cost)
        return vertex_id, cost

    def update(self, vertex_id: VertexId, cost: Cost) -> None:
        # Base case: New vertex
        if vertex_id not in self.cost:
            self.cost[vertex_id] = cost
            if cost not in self.index:
                self.index[cost] = set()
                self.keys.add(cost)
            self.index[cost].add(vertex_id)

        # Recursive case: Existing vertex
        else:
            previous_cost = self.cost[vertex_id]
            if cost == previous_cost:
                return  # Do nothing

            # Remove from data structures
            del self.cost[vertex_id]
            self.index[previous_cost].remove(vertex_id)
            if len(self.index[previous_cost]) == 0:
                del self.index[previous_cost]
                self.keys.remove(previous_cost)

            # Add new cost by recursion
            self.update(vertex_id=vertex_id, cost=cost)


class Graph:
    def __init__(self):
        self.vertices: dict[VertexId, Vertex] = dict()

    def add_vertex(self, vertex_id: VertexId) -> None:
        self.vertices[vertex_id] = Vertex(id=vertex_id, out_edges=set())

    def add_edge(self, edge: Edge) -> None:
        from_vertex = self.vertices[edge.from_vertex_id]
        from_vertex.out_edges.add(edge)

    def single_source_shortest_path(self, source_vertex_id: VertexId) -> dict[VertexId, VertexId]:
        """
        Generates all shortest paths from source.
        For this project, this computation could be cached for speed improvements.

        :param source_vertex_id:
        :return: A reverse dictionary from destinations back to the specified source
        """
        vertex_queue = SortedIndexDict((vertex_id, inf)
                                       for vertex_id in self.vertices.keys())
        vertex_queue.update(source_vertex_id, 0)
        parent_vertices: dict[VertexId, VertexId] = dict()
        parent_vertices[source_vertex_id] = source_vertex_id

        while len(vertex_queue) > 0:
            source_vertex_id, source_vertex_cost = vertex_queue.pop_lowest()
            source_vertex = self.vertices[source_vertex_id]
            for edge in source_vertex.out_edges:
                to_vertex_id = edge.to_vertex_id
                total_edge_cost = source_vertex_cost + edge.cost
                try:
                    to_vertex_cost = vertex_queue.get_cost(to_vertex_id)
                except KeyError:
                    pass  # We have already visited this vertex
                else:
                    if total_edge_cost < to_vertex_cost:  # Relaxation step
                        vertex_queue.update(to_vertex_id, total_edge_cost)
                        parent_vertices[to_vertex_id] = source_vertex_id

        return parent_vertices

    def shortest_path(self, source_vertex_id: VertexId, to_vertex_id: VertexId) -> Iterable[VertexId]:
        """
        Traverses the path from the destination back to the source.

        :param source_vertex_id: The source location
        :param to_vertex_id: The destination
        :return: A generator of each step along the way
        """
        shortest_path_map = self.single_source_shortest_path(source_vertex_id=source_vertex_id)

        yield to_vertex_id
        next_vertex_id = to_vertex_id
        while next_vertex_id != source_vertex_id:
            next_vertex_id = shortest_path_map[next_vertex_id]
            yield next_vertex_id


def _main():
    """
    Set up facts in project description
    """

    graph = Graph()
    vertices = (
        'CAN',
        'USA',
        'MEX',
        'BLZ',
        'GTM',
        'SLV',
        'HND',
        'NIC',
        'CRI',
        'PAN',
    )
    for vertex in vertices:
        graph.add_vertex(vertex)

    edges = (
        # Canada borders the United States
        ('CAN', 'USA'),

        # The United States borders Canada and Mexico
        ('USA', 'MEX'),

        # Mexico borders the United States, Guatemala, and Belize
        ('MEX', 'BLZ'),
        ('MEX', 'GTM'),

        # Belize borders Mexico and Guatemala
        ('BLZ', 'GTM'),

        # Guatemala borders Mexico, Belize, El Salvador, and Honduras
        ('GTM', 'SLV'),
        ('GTM', 'HND'),

        # El Salvador borders Guatemala and Honduras
        ('SLV', 'HND'),

        # Honduras borders Guatemala, El Salvador, and Nicaragua
        ('HND', 'NIC'),

        # Nicaragua borders Honduras and Costa Rica
        ('NIC', 'CRI'),

        # Costa Rica borders Nicaragua and Panama
        ('CRI', 'PAN'),
    )
    for from_vertex_id, to_vertex_id in edges:
        graph.add_edge(Edge(from_vertex_id=from_vertex_id, to_vertex_id=to_vertex_id, cost=1))
        # Edges are reciprocal
        graph.add_edge(Edge(from_vertex_id=to_vertex_id, to_vertex_id=from_vertex_id, cost=1))

    print(list(reversed(list(graph.shortest_path('USA', 'PAN')))))


if __name__ == '__main__':
    _main()
