# Change Log
All notable changes to this project will be documented in this file.

## [0.0.4] - 2014-12-15
### Added
- MathModes enum for selecting fast operations at the expense of precision

### Changed
- expit combines all the operations into one kernel for speed boost

## [0.0.3] - 2014-12-14
### Added
- GPU accelerated expit function
- Utility functions for switching between GPUArray and numpy array
- Decorator to manage input and output types for gpu functions

### Changed
- correlate can accept both GPUArray and numpy arrays as input
- enums are now defined in sciguppy.enums

## [0.0.1] - 2014-11-26
### Added
- GPU accelerated correlation function
