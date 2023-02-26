[![Worflows](https://github.com/jasonrhansen/cut-optimizer-1d/actions/workflows/rust.yml/badge.svg)](https://github.com/jasonrhansen/cut-optimizer-1d/actions)
[![Crates.io](https://img.shields.io/crates/v/cut-optimizer-1d.svg)](https://crates.io/crates/cut-optimizer-1d)
[![Documentation](https://docs.rs/cut-optimizer-1d/badge.svg)](https://docs.rs/cut-optimizer-1d/)
[![Dependency status](https://deps.rs/repo/github/jasonrhansen/cut-optimizer-1d/status.svg)](https://deps.rs/repo/github/jasonrhansen/cut-optimizer-1d)

# Cut Optimizer 1D

## Description

Cut Optimizer 1D is a cut optimizer library for optimizing linear cuts.

Given desired cut pieces and stock pieces, it will attempt to layout the cut
pieces in way that gives the least waste.
It can't guarantee the most optimizal solution possible, since this would be too
inefficient. Instead it uses genetic
algorithms and multiple heuristics to solve the problem. This usually results in
a satisfactory solution.

## License

Duel-license under MIT license ([LICENSE-MIT](LICENSE-MIT)), or Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
