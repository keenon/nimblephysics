# Devnote for DiffDART

## Changelog

DiffDART was forked from [dartsim/dart#514a928](https://github.com/dartsim/dart/tree/514a92800dd9f0dc4c2587b97e5b04e1de0e3fe3).

The following is the incomplete changelog of DiffDART. [A], [M], and [R] denote added, modified, and removed, respectively.

- [M] dart
  - [M] simulation
    - [M] World.hpp
      - [A] World::getPositions()
      - [A] World::getVelocities()
      - [A] World::getAccelerations()
- [M] unittests
  - [M] comprehensive
    - [A] test_ParallelOps.cpp
  - [M] unit
    - [A] test_ScrewGeometry.cpp
