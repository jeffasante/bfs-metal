// Sources/BFSMetal/Graph.swiftimport Foundation

/// Represents a graph using adjacency list representation
struct Graph {
    /// Adjacency list representation of the graph
    private var adjacencyList: [[Int]]
    
    /// Number of vertices in the graph
    var vertexCount: Int {
        return adjacencyList.count
    }
    
    /// Initialize an empty graph with a given number of vertices
    init(vertexCount: Int) {
        self.adjacencyList = Array(repeating: [], count: vertexCount)
    }
    
    /// Add an edge to the graph
    mutating func addEdge(from: Int, to: Int) {
        guard from >= 0 && from < vertexCount && to >= 0 && to < vertexCount else {
            return
        }
        adjacencyList[from].append(to)
    }
    
    /// Get neighbors of a vertex
    func neighbors(of vertex: Int) -> [Int] {
        guard vertex >= 0 && vertex < vertexCount else {
            return []
        }
        return adjacencyList[vertex]
    }
    
    /// Convert to Compressed Sparse Row (CSR) format for GPU processing ( in our case for metal shader)
    func toCSR() -> (offsets: [Int], edges: [Int]) {
        var offsets = [0] // An array indicating the start index of neighbors for each vertex.
        var edges: [Int] = [] // A flat array storing all neighbors.
        
        for vertices in adjacencyList {
            edges.append(contentsOf: vertices)
            offsets.append(offsets.last! + vertices.count)
        }
        
        return (offsets, edges)
    }
}