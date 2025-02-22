//Tests/BFSMetalTests/BFSTests.swift

import XCTest
import Metal
@testable import BFSMetal

final class BFSTests: XCTestCase {
    // Check if Metal is available
    var isMetalAvailable: Bool {
        MTLCreateSystemDefaultDevice() != nil
    }
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        // Skip all tests if Metal is not available
        if !isMetalAvailable {
            throw XCTSkip("Skipping tests - Metal is not available on this system")
        }
    }

    func testMetalBFS() throws {
        // Create our test graph:
        // 0 --- 1 --- 3 --- 5
        // |     |
        // 2 --- 4
        var graph = Graph(vertexCount: 6)
        
        // Add edges (making it undirected by adding both directions)
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
        
        // Expected distances from vertex 0
        let expectedDistances = [0, 1, 1, 2, 2, 3]
        
        // Run Metal BFS
        let metalBFS = try MetalBFS()
        let distances = try metalBFS.traverse(graph: graph, source: 0)
        
        // Verify results
        XCTAssertEqual(distances, expectedDistances, "Metal BFS distances do not match expected values")
    }
    
    func testCPUvsGPUBFS() throws {
        var graph = Graph(vertexCount: 6)
        
        // Same test graph setup
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
        
        // Run CPU BFS
        let cpuBFS = BFS(graph: graph)
        let cpuResult = cpuBFS.traverse(from: 0)
        
        // Run GPU BFS
        let metalBFS = try MetalBFS()
        let gpuDistances = try metalBFS.traverse(graph: graph, source: 0)
        
        // Compare results
        XCTAssertEqual(cpuResult.distances, gpuDistances, "CPU and GPU BFS results don't match")
    }
}