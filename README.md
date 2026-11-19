# CUDA Rigidbody Simulation

This project implements a basic rigidbody simulation of 2D balls including gravity, collisions, friction, and bounce. To achieve high performance with millions of objects, the world is partitioned into a grid where each cell tracks its containing objects. This allows efficient lookup of nearby objects for collisions, which would otherwise be an O(n^2) algorithm. Additionally, I use CUDA to draw directly to a Raylib texture to render objects, which means drawing performance scales with the number of pixels rather than the number of objects. If I were to have used meshes instead, performance would not scale up to millions (even using instanced meshes).

## Dependencies

- CUDA
- cuRAND
- [Raylib](https://www.raylib.com/)

I have included a Nix flake for convenience. Otherwise, you will have to install Raylib yourself: https://repology.org/project/raylib/versions

## Building

```sh
make && ./main
```

For simplicity's sake, I kept the code in one file, `main.cu`. It compiles in less than two seconds on my machine with optimizations enabled.
