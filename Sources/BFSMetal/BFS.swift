import Foundation

/// BFS result containing distances and parent information
struct BFSResult {
    /// Distance from source to each vertex (-1 if unreachable)
    let distances: [Int] // An array storing the shortest distance from the source vertex to each vertex.
    /// Parent of each vertex in BFS tree (-1 for unreachable vertices)
    let parents: [Int] // An array storing the parent vertex of each vertex in the BFS tree.
}

class BFS {
    private let graph: Graph
    
    init(graph: Graph) {
        self.graph = graph
    }
    
    /// Perform sequential BFS from a source vertex
    /// - Parameter source: Starting vertex for BFS
    /// - Returns: BFS result containing distances and parents
    func traverse(from source: Int) -> BFSResult {
        let vertexCount = graph.vertexCount
        var distances = Array(repeating: -1, count: vertexCount)
        var parents = Array(repeating: -1, count: vertexCount)
        var queue = Queue<Int>()
        
        // Initialize source vertex
        distances[source] = 0
        queue.enqueue(source)
        
        // BFS main loop
        while let current = queue.dequeue() {
            // Process all neighbors
            for neighbor in graph.neighbors(of: current) {
                // If neighbor not visited (distance == -1)
                if distances[neighbor] == -1 {
                    distances[neighbor] = distances[current] + 1 // update its distance,
                    parents[neighbor] = current
                    queue.enqueue(neighbor)
                }
            }
        }
        
        return BFSResult(distances: distances, parents: parents)
    }
}

// MARK: - Queue Implementation
private struct Queue<T> { // 
    private var array: [T] = []
    
    mutating func enqueue(_ element: T) {
        array.append(element)
    }
    
    mutating func dequeue() -> T? {
        guard !array.isEmpty else { return nil }
        return array.removeFirst()
    }
    
    var isEmpty: Bool {
        return array.isEmpty
    }
}

// MARK: - BFS Result Extensions
extension BFSResult {
    /// Get the path from source to a target vertex
    /// - Parameter target: Target vertex to find path to
    /// - Returns: Array of vertices in the path, or empty if no path exists
    func path(to target: Int) -> [Int] {
        guard parents[target] != -1 else { return [] }
        
        var path = [target]
        var current = target
        
         while parents[current] != -1 {
            current = parents[current]
            path.append(current)
        }
        
        return path.reversed()
    }
    
    /// Print BFS statistics
    func printStats() {
        let reachableCount = distances.filter { $0 != -1 }.count
        let maxDistance = distances.max() ?? -1
        
        print("BFS Statistics:")
        print("- Reachable vertices: \(reachableCount)")
        print("- Maximum distance: \(maxDistance)")
    }
}