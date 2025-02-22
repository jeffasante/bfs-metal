// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "BFSMetal",
    platforms: [
       .macOS(.v12)
    ],
    products: [
        .executable(name: "BFSMetal", targets: ["BFSMetal"])
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "BFSMetal",
            dependencies: [],
            resources: [
                 .copy("Shaders/BFSShaders.metal")
            ]
        ),
        .testTarget(
            name: "BFSMetalTests",
            dependencies: ["BFSMetal"]
        )
    ]
)