# Changelog

All notable changes to the HBM System Modeling Platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Logic Base Die model (`logic_base_die_example.py`)
  - 32-channel independent per-channel operations
  - PAM3 encoding support
  - Command buffer management with priority queuing
  - Per-channel timing with independent cycle management
  - Calibration data storage and retrieval
- EXAMPLES.md with 9 comprehensive usage examples
- Complete README.md with feature list and test results

### Fixed
- Python 3.10 compatibility (removed dataclass slots=True)

## [1.0.0] - 2026-06-16

### Added

#### HBM4 Core Architecture
- 32-channel architecture (2x HBM3 channel count)
- Pseudo-channel support (64 total pseudo-channels)
- Speed grades: 8 Gbps, 12 Gbps, 16 Gbps
- 2048-bit I/O width with 2 TB/s peak bandwidth
- 8 bank groups, 16 banks per group
- 64K rows per bank, 256 columns per row
- 2KB row buffer optimization

#### Controller Components (Phase A)
- `HBM4Controller` - Complete controller integration
- `HBM4AddressDecoder` - 32-channel address decoding with RBC/BCR/CRB mapping
- `HBM4QoSScheduler` - 16-level priority scheduling with anti-starvation
- `HBM4RefreshScheduler` - All-bank, per-bank, and autonomous refresh modes
- DRFM (Direct Refresh Management) for row-hammer mitigation
- Request queue management with `QueueManager`

#### DRAM Model (Phase B)
- `HBM4Spec` - Complete HBM4 specification constants
- `HBM4ChannelModel` / `HBM4ChannelArray` - 32-channel DRAM model
- `BankStateMachine` - Per-bank state tracking (IDLE, ACTIVE, etc.)
- `DFI5Interface` - Full DFI 5.0/5.1 protocol implementation
- `PHYTrainer` - PHY initialization and training sequences
- `LaneRepairMapper` - Redundant lane mapping and repair
- `ECCEngine` and `CRCGenerator` - Error detection and correction
- `PowerEstimator` - Power and energy estimation
- `MBISTController` - Memory built-in self-test
- `LoopbackController` - Loopback testing for PHY verification
- `LogicBaseDie` - Complete logic base die model

#### RTL Implementation (Phase C)
- `hbm_types.svh` - SystemVerilog type definitions
- `hbm_controller.sv` - RTL controller implementation
- `dram_model.sv` - Behavioral DRAM model with timing compliance
- `hbm_pkg.sv` - UVM package definitions
- `hbm_controller_tb.cpp` - C++ testbench

#### UVM Verification (Phase D)
- Complete UVM testbench environment
- Reference models for Python-RTL co-simulation
- Functional coverage collection
- CI/CD pipeline with GitHub Actions

#### Testing Infrastructure
| Category | Tests | Status |
|----------|-------|--------|
| Controller Tests | 130 | Passing |
| DRAM Tests | 612 | Passing |
| HBM4 DFI Tests | 99 | Passing |
| HBM4 PHY/TSV/Lane | 534 | Passing |
| Simulation Tests | 156 | Passing |
| Integration Tests | 583 | Passing |
| Coverage Tests | 354 | Passing |
| Benchmark Tests | 381 | Passing |
| **Total** | **2849** | **All Passing** |

#### Examples
- `basic_controller.py` - Basic controller creation and request submission
- `address_decoding.py` - Address decoding with RBC/BCR/CRB mapping schemes
- `qos_scheduling.py` - 16-level QoS priority scheduling with anti-starvation
- `refresh_scheduling.py` - All-bank, per-bank, and DRFM refresh modes
- `dfi_interface.py` - DFI 5.0 interface operations and protocols
- `bandwidth_benchmark.py` - Bandwidth measurement and performance benchmarking
- `multi_channel.py` - 32-channel multi-channel operations
- `dram_features.py` - ECC/CRC, lane repair, PHY training, MBIST
- `logic_base_die_example.py` - Logic base die comprehensive example

#### Documentation
- Complete README.md with feature list
- EXAMPLES.md with 9 comprehensive usage examples
- CHANGELOG.md with version history
- Design document with complete HBM4 specification
- HBM3 specification reference

### Fixed
- Python 3.10 compatibility - removed dataclass `slots=True`
- Benchmark test failures
- Integration test sequences
- Address code review issues in benchmark module

### Changed
- Updated README.md with comprehensive HBM4 feature list
- Improved documentation structure and organization
- Improved controller architecture for better modularity
- Enhanced DFI interface timing compliance
- Optimized refresh scheduling for reduced power
- Updated bandwidth calculation formulas

## [0.9.0] - 2026-06-15

### Added
- Initial HBM system modeling platform
- Phase A HBM Controller Model core framework
- Phase B DRAM timing model
- Basic test infrastructure
- Project documentation

### Features
- HBM3-compatible base controller
- Basic address decoding
- Simple request queue
- Bank state machine
- Channel model

---

## Migration Guide

### Upgrading to 1.0.0

#### HBM4 Controller
The HBM4 controller is now the default:

```python
from model.controller.hbm4_controller import HBM4Controller
controller = HBM4Controller()
```

#### Address Decoding
Address decoder now supports multiple mapping schemes:

```python
decoder = HBM4AddressDecoder(mapping_scheme="rbc")  # Row-Bank-Channel
decoder = HBM4AddressDecoder(mapping_scheme="bcr")   # Bank-Channel-Row
decoder = HBM4AddressDecoder(mapping_scheme="crb")   # Channel-Row-Bank
```

#### DFI Interface
DFI interface version is now 5.0:

```python
from model.dram.dfi_interface import DFI5Interface
dfi = DFI5Interface()  # Default DFI 5.0
```

---

## Deprecations

### HBM3 Controller (Deprecated)
The HBM3 controller model is deprecated and will be removed in 2.0.0.
Use `HBM4Controller` with appropriate parameters instead.

### Old DFI Interface (Deprecated)
The DFI 4.0 interface is deprecated. Use `DFI5Interface` instead.

---

[Unreleased]: https://github.com/anthropic/hbm/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/anthropic/hbm/releases/tag/v1.0.0
[0.9.0]: https://github.com/anthropic/hbm/releases/tag/v0.9.0