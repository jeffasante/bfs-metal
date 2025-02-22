// Sources/BFSMetal/MetalBFS.swift
#if os(macOS)
import Foundation
import Metal

@available(macOS 12.0, *)
class MetalBFS {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState
    
    private struct BFSUniforms {
        var vertexCount: UInt32
        var level: UInt32
        var edgeCount: UInt32
    }
    
    enum MetalBFSError: Error {
        case deviceNotFound
        case commandQueueCreationFailed
        case libraryCreationFailed
        case functionNotFound
        case pipelineCreationFailed
        case bufferCreationFailed
    }
    
    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalBFSError.deviceNotFound
        }
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalBFSError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue
        
        do {
            // Load shader code
            guard let shaderURL = Bundle.module.url(forResource: "BFSShaders", withExtension: "metal") else {
                print("Failed to find BFSShaders.metal in bundle")
                throw MetalBFSError.libraryCreationFailed
            }
            
            let metalLibrary = try String(contentsOf: shaderURL, encoding: .utf8)
            print("Loaded shader source successfully")
            
            let compileOptions = MTLCompileOptions()
            compileOptions.languageVersion = .version2_4
            
            let library = try device.makeLibrary(source: metalLibrary, options: compileOptions)
            
            guard let bfsFunction = library.makeFunction(name: "bfs_kernel") else {
                throw MetalBFSError.functionNotFound
            }
            
            pipelineState = try device.makeComputePipelineState(function: bfsFunction)
        } catch {
            print("Failed to create library: \(error)")
            throw MetalBFSError.libraryCreationFailed
        }
    }
    
    func traverse(graph: Graph, source: Int) throws -> [Int] {
        let (offsets, edges) = graph.toCSR()
        
        // Initialize distances array
        var distances = Array(repeating: Int32.max, count: graph.vertexCount)
        distances[source] = 0
        
        var changedCounter: UInt32 = 0
        
        // Convert to Int32 for Metal
        let offsetsInt32 = offsets.map { Int32($0) }
        let edgesInt32 = edges.map { Int32($0) }
        
        // Create Metal buffers with optimal settings
        guard let distanceBuffer = device.makeBuffer(bytes: &distances,
                                                   length: MemoryLayout<Int32>.stride * graph.vertexCount,
                                                   options: [.storageModeShared, .cpuCacheModeWriteCombined]),
              let edgesBuffer = device.makeBuffer(bytes: edgesInt32,
                                                length: MemoryLayout<Int32>.stride * edges.count,
                                                options: [.storageModeShared, .cpuCacheModeWriteCombined]),
              let offsetsBuffer = device.makeBuffer(bytes: offsetsInt32,
                                                  length: MemoryLayout<Int32>.stride * offsets.count,
                                                  options: [.storageModeShared, .cpuCacheModeWriteCombined]),
              let changedBuffer = device.makeBuffer(bytes: &changedCounter,
                                                  length: MemoryLayout<UInt32>.size,
                                                  options: [.storageModeShared, .cpuCacheModeWriteCombined]) else {
            throw MetalBFSError.bufferCreationFailed
        }
        
        // Main BFS loop
        for level in 0..<graph.vertexCount {
            changedCounter = 0
            changedBuffer.contents().copyMemory(from: &changedCounter, byteCount: MemoryLayout<UInt32>.size)
            
            var uniforms = BFSUniforms(
                vertexCount: UInt32(graph.vertexCount),
                level: UInt32(level),
                edgeCount: UInt32(edges.count)
            )
            
            guard let uniformBuffer = device.makeBuffer(bytes: &uniforms,
                                                      length: MemoryLayout<BFSUniforms>.stride,
                                                      options: [.storageModeShared, .cpuCacheModeWriteCombined]) else {
                throw MetalBFSError.bufferCreationFailed
            }
            
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalBFSError.commandQueueCreationFailed
            }
            
            // Optimize thread group size
            let threadsPerGroup = min(pipelineState.maxTotalThreadsPerThreadgroup, 256)
            let threadGroupCount = (graph.vertexCount + threadsPerGroup - 1) / threadsPerGroup
            
            // 
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(distanceBuffer, offset: 0, index: 0)
            encoder.setBuffer(edgesBuffer, offset: 0, index: 1)
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 2)
            encoder.setBuffer(changedBuffer, offset: 0, index: 3)
            encoder.setBuffer(uniformBuffer, offset: 0, index: 4)
            
            encoder.dispatchThreadgroups(
                MTLSize(width: threadGroupCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
            
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            changedCounter = changedBuffer.contents().load(as: UInt32.self)
            if changedCounter == 0 {
                break
            }
        }
        
        let finalDistances = Array(UnsafeBufferPointer(
            start: distanceBuffer.contents().assumingMemoryBound(to: Int32.self),
            count: graph.vertexCount
        )).map { $0 == Int32.max ? -1 : Int($0) }
        
        return finalDistances
    }
}
#endif