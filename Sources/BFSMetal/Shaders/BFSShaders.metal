#include <metal_stdlib>
using namespace metal;

struct BFSUniforms {
    uint vertex_count;
    uint level;
    uint edge_count;
};

kernel void bfs_kernel(
    device atomic_int* distances [[buffer(0)]],
    device const int* edges [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    device atomic_uint* changed [[buffer(3)]],
    constant BFSUniforms& uniforms [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]
) {
    // Process multiple vertices per thread for better utilization
    for (uint vid = tid; vid < uniforms.vertex_count; vid += threads_per_grid) {
        // Check if current vertex is at the current level
        if (atomic_load_explicit((device atomic_int*)&distances[vid], memory_order_relaxed) == uniforms.level) {
            // Get edge range for current vertex
            uint start_edge = offsets[vid];
            uint end_edge = (vid + 1 < uniforms.vertex_count) ? offsets[vid + 1] : uniforms.edge_count;
            
            // Pre-calculate next level
            int next_level = uniforms.level + 1;
            
            // Process edges in chunks for better memory coalescing
            for (uint i = start_edge; i < end_edge; i++) {
                uint neighbor = edges[i];
                int expected = INT_MAX;
                
                // Optimized atomic operation
                if (atomic_compare_exchange_weak_explicit(
                    (device atomic_int*)&distances[neighbor],
                    &expected,
                    next_level,
                    memory_order_relaxed,
                    memory_order_relaxed)) {
                    atomic_fetch_add_explicit(changed, 1, memory_order_relaxed);
                }
            }
        }
    }
}