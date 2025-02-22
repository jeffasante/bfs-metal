// Sources/BFSMetal/main.swift

import Foundation

// Create a sample graph
var graph = Graph(vertexCount: 6)

// Add edges to create this graph:
//     0 --- 1 --- 3 --- 5
//     |     |
//     2 --- 4
graph.addEdge(from: 0, to: 1)
graph.addEdge(from: 0, to: 2)
graph.addEdge(from: 1, to: 0)
graph.addEdge(from: 1, to: 3)
graph.addEdge(from: 1, to: 4)
graph.addEdge(from: 2, to: 0)
graph.addEdge(from: 2, to: 4)
graph.addEdge(from: 3, to: 1)
graph.addEdge(from: 3, to: 5)
graph.addEdge(from: 4, to: 1)
graph.addEdge(from: 4, to: 2)
graph.addEdge(from: 5, to: 3)

// Create BFS instance
let bfs = BFS(graph: graph)

// Run BFS from vertex 0
print("Running BFS from vertex 0...")
let result = bfs.traverse(from: 0)

// Print results
print("\nDistances from source:")
for (vertex, distance) in result.distances.enumerated() {
    print("Vertex \(vertex): \(distance)")
}

print("\nPath from 0 to 5:")
let pathTo5 = result.path(to: 5)
print(pathTo5.map(String.init).joined(separator: " -> "))

// Print statistics
print("\nStatistics:")
result.printStats()