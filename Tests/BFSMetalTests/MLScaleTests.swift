import XCTest
import Metal
@testable import BFSMetal

final class MLScaleTests: XCTestCase {
    // Large graph parameters
    let vertexCount = 100000  // 100K vertices
    let averageConnections = 50  // Each vertex connects to ~50 others
    
    func testLargeScaleGraphPerformance() throws {
        // Create a large random graph that mimics a ML similarity network
        var graph = Graph(vertexCount: vertexCount)
        var edgeCount = 0
        
        // Generate random connections with locality preference
        // (mimicking how ML embeddings often have local structure)
        for vertex in 0..<vertexCount {
            // Create local connections (nearby vertices are more likely to connect)
            let localConnections = Int.random(in: averageConnections/2...averageConnections*3/2) // generates a random integer between a and b, inclusive.
            
            for _ in 0..<localConnections {
                // Generate connections with locality bias
                let maxDistance = min(1000, vertexCount/4)  // Local neighborhood size
                let offset = Int.random(in: -maxDistance...maxDistance)
                let target = (vertex + offset + vertexCount) % vertexCount
                
                if target != vertex {
                    graph.addEdge(from: vertex, to: target)
                    edgeCount += 1
                }
            }
            
            // Add some random long-distance connections (small world property)
            if Double.random(in: 0...1) < 0.1 {  // 10% chance
                let target = Int.random(in: 0..<vertexCount)
                if target != vertex {
                    graph.addEdge(from: vertex, to: target)
                    edgeCount += 1
                }
            }
        }
        
        print("\nGraph Statistics:")
        print("Vertices: \(vertexCount)")
        print("Total Edges: \(edgeCount)")
        print("Average Degree: \(Double(edgeCount)/Double(vertexCount))")
        
        // Measure CPU BFS performance
        let cpuStartTime = CFAbsoluteTimeGetCurrent()
        let cpuBFS = BFS(graph: graph)
        let cpuResult = cpuBFS.traverse(from: 0)
        let cpuEndTime = CFAbsoluteTimeGetCurrent()
        let cpuTime = cpuEndTime - cpuStartTime
        
        // Measure GPU BFS performance
        let metalBFS = try MetalBFS()
        let gpuStartTime = CFAbsoluteTimeGetCurrent()
        let gpuDistances = try metalBFS.traverse(graph: graph, source: 0)
        let gpuEndTime = CFAbsoluteTimeGetCurrent()
        let gpuTime = gpuEndTime - gpuStartTime
        
        // Verify results match
        XCTAssertEqual(cpuResult.distances, gpuDistances, "CPU and GPU results should match")
        
        // Calculate and print performance metrics
        print("\nPerformance Comparison:")
        print("CPU Time: \(String(format: "%.3f", cpuTime)) seconds")
        print("GPU Time: \(String(format: "%.3f", gpuTime)) seconds")
        print("Speedup: \(String(format: "%.2fx", cpuTime/gpuTime))")
        
        // Calculate connectivity statistics
        let reachableFromSource = cpuResult.distances.filter { $0 != -1 }.count
        let maxDistance = cpuResult.distances.max() ?? -1
        
        print("\nTraversal Statistics:")
        print("Reachable vertices: \(reachableFromSource) (\(String(format: "%.1f", Double(reachableFromSource)/Double(vertexCount) * 100))%)")
        print("Maximum distance from source: \(maxDistance)")
        
        // Verify reasonable connectivity
        XCTAssertGreaterThan(
            Double(reachableFromSource)/Double(vertexCount),
            0.9,  // At least 90% should be reachable
            "Graph should be well-connected"
        )
        
        // Verify performance improvement
        XCTAssertLessThan(
            gpuTime,
            cpuTime,
            "GPU implementation should be faster than CPU for large graphs"
        )
    }
}