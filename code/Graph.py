class Graph:

    class Vertex:

        def __init__(self, value, edges = []):
            self.value = value
            self.edges = edges

        def get_edges(self):
            return self.edges
        
        def get_value(self):
            return self.value
        
        def add_edge(self, edge):
            self.edges.append(edge)

    class Edge:

        def __init__(self, u, v, dist):
            self.u = u
            self.v = v
            self.dist = dist

        def get_u(self):
            return self.u
        
        def get_v(self):
            return self.v
        
        def get_dist(self):
            return self.dist
        
        def set_dist(self, dist):
            self.dist = dist

    def __init__(self, vertices, edges):
        
        # create vertices
        self.vertices = {}
        for vertex in vertices:
            self.vertices[vertex] = self.Vertex(vertex)
        
        # create edges
        for edge in edges:
            self.vertices[edge[0]].add_edge(self.Edge(edge[0], edge[1], edge[2]))
            self.vertices[edge[1]].add_edge(self.Edge(edge[0], edge[1], edge[2]))

    # get the vertices of the graph
    def get_vertices(self):
        return self.vertices
    
    # get a specific vertex of the graph
    def get_vertex(self, value):
        return self.vertices[value]
    
    # get the edges of a vertex
    def get_edges(self, vertex):
        return self.vertices[vertex].get_edges()
    


    