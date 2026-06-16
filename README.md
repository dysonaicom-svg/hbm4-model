# HBM System Modeling Platform

High Bandwidth Memory (HBM) system simulation platform for chip design exploration and verification alignment.

## Features

- **HBM3 and HBM4 Support** - Full controller, DRAM model, DFI interface, and PHY-level simulation
- **32-Channel Architecture** - 2x HBM3 channel count for increased bandwidth
- **Speed Grades** - 8 Gbps, 12 Gbps, 16 Gbps data rates
- **2 TB/s Peak Bandwidth** - Aggregate bandwidth at 16 Gbps
- **Multi-Channel Load Balancing** - Adaptive channel selection and fair bandwidth distribution
- **Signal Integrity Analysis** - TX pre-emphasis, RX CTLE, DFE, IBIS models, eye diagram analysis
- **DFI 5.0 Interface** - Complete controller-PHY interface specification

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install as editable package (recommended)
pip install -e .
```

### Basic Usage

```python
from model.controller.hbm4_controller import HBM4Controller

# Create controller
controller = HBM4Controller()

# Submit read request
request_id = controller.submit_request(
    addr=0x0001_0000_0000_0000,
    is_read=True,
    qos_level=8,
)

# Run simulation
for _ in range(100):
    responses = controller.tick()
    for resp in responses:
        print(f"Completed: {resp.request_id}")

# Get statistics
stats = controller.get_stats()
print(f"Row hit rate: {stats['controller']['row_hit_rate']:.2%}")
```

### Multi-Channel Load Balancing

```python
from model.multi_channel import MultiChannelTrafficGenerator, ChannelSelector

# Create channel selector with adaptive load balancing
channel_selector = ChannelSelector(
    num_channels=32,
    strategy=ChannelSelector.ADAPTIVE
)

# Create traffic generator
traffic_gen = MultiChannelTrafficGenerator(
    config=sim_config,
    num_channels=32,
    channel_selector=channel_selector
)

# Generate requests with balanced channel distribution
requests = traffic_gen.generate_burst()

# Check load balance metrics
jain_fairness = simulator.get_jains_fairness_index()
print(f"Jain's fairness index: {jain_fairness:.3f}")
```

### Signal Integrity Analysis

```python
from model.phy.signal_integrity import SignalIntegrityConfig, TXPreEmphasis, RXCTLE
from model.phy.eye_analyzer import EyeAnalyzer

# Create signal integrity components
config = SignalIntegrityConfig(sample_rate=32e9, ui_ns=0.125)
tx = TXPreEmphasis()
rx_ctle = RXCTLE()

# Configure pre-emphasis
tx.set_taps([-0.2, 1.0, -0.1])  # Pre, main, post cursor

# Generate eye diagram
eye_data = tx.estimate_tx_eye(prbs_length=1024)

# Analyze eye metrics
analyzer = EyeAnalyzer()
metrics = analyzer.analyze_eye(eye_data)
print(f"Eye width: {metrics.eye_width:.3f} UI")
print(f"Eye height: {metrics.eye_height:.3f} V")
```

### Full System Simulation

```python
from sim.simulator import HBMSimulator, SimulationConfig, TrafficPattern

# Create simulation configuration
config = SimulationConfig(
    simulation_time_us=100.0,
    traffic_pattern=TrafficPattern.RANDOM,
    request_rate=0.5,
    read_ratio=0.7,
    max_requests_per_cycle=4,
)

# Create and run simulator
sim = HBMSimulator(config)
stats = sim.run()

# Get results
print(f"Throughput: {stats.throughput_gbps:.2f} GB/s")
print(f"Row hit rate: {stats.row_hit_rate:.2%}")
print(f"Efficiency: {stats.efficiency:.2%}")
```

## Architecture

### System Architecture

```
                    +-----------------------------+
                    |   Traffic Generator /       |
                    |   Trace Reader              |
                    +-------------+---------------+
                                  |
                                  v
                    +-----------------------------+
                    |   Multi-Channel Load       |
                    |   Balancer                  |
                    +-------------+---------------+
                                  |
                                  v
                    +-----------------------------+
                    |   HBM Controller            |
                    |   - Address Decoder         |
                    |   - Request Queue           |
                    |   - FR-FCFS / QoS Scheduler |
                    |   - Refresh Scheduler       |
                    +-------------+---------------+
                                  |
                    +-------------+---------------+
                    |          DFI Interface       |
                    |         (DFI 5.0/5.1)       |
                    +-------------+---------------+
                                  |
                    +-----------------------------+
                    |   HBM DRAM Model            |
                    |   - Channel Model           |
                    |   - Bank State Machine      |
                    |   - PHY Training            |
                    |   - ECC/CRC                 |
                    |   - Lane Repair             |
                    +-------------+---------------+
                                  |
                    +-----------------------------+
                    |   Signal Integrity Module   |
                    +-------------+---------------+
                                  |
                                  v
                    +-----------------------------+
                    |   Statistics Collector      |
                    +-----------------------------+
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| HBM4 Controller | `model/controller/hbm4_controller.py` | Main controller with QoS/FR-FCFS |
| Address Decoder | `model/controller/hbm4_address_decoder.py` | RBC/BCR/CRB address mapping |
| QoS Scheduler | `model/controller/hbm4_qos_scheduler.py` | 16-level priority scheduling |
| Refresh Scheduler | `model/controller/hbm4_refresh_scheduler.py` | All-bank/per-bank refresh |
| DFI Interface | `model/dram/dfi_interface.py` | DFI 5.0/5.1 protocol |
| Channel Model | `model/dram/hbm4_channel_model.py` | Per-channel timing model |
| Bank State Machine | `model/dram/bank_state_machine.py` | Per-bank state tracking |
| Lane Repair | `model/dram/lane_repair.py` | Redundant lane mapping |
| ECC/CRC | `model/dram/ecc_crc.py` | Error detection/correction |
| PHY Training | `model/dram/phy_training.py` | Calibration sequences |
| Signal Integrity | `model/phy/signal_integrity.py` | TX/RX equalization |
| Eye Analyzer | `model/phy/eye_analyzer.py` | Eye diagram metrics |
| Interconnect | `sim/interconnect/` | AXI crossbar and NoC mesh |
| Trace Parser | `sim/trace/` | External trace replay |

## Running Tests

```bash
# All tests
pytest tests/ -v

# By category
pytest tests/controller/ -v    # Controller tests
pytest tests/dram/ -v          # DRAM tests
pytest tests/hbm4/ -v          # HBM4 tests
pytest tests/phy/ -v            # PHY/Signal Integrity tests
pytest tests/integration/ -v    # Integration tests
pytest tests/benchmark/ -v     # Benchmark tests

# Specific module tests
pytest tests/hbm4/test_pam3.py -v           # PAM3 encoding tests
pytest tests/hbm4/test_logic_base_die.py -v # LBD integration tests
pytest tests/hbm4/test_channel_timing.py -v # Channel timing tests
```

## RTL Simulation

```bash
# Compile RTL with Verilator
cd rtl && verilator --cc --trace hbm_controller.sv hbm_types.svh

# Run simulation
cd rtl && make sim

# Lint check
cd rtl && make lint

# Build with waveform
cd rtl && make sim-debug
```

## Troubleshooting

### Queue Full Errors

If `submit_request()` returns `None`, the request queue is full:

```python
# Check queue depth before submitting
stats = controller.get_stats()
if stats['queues']['read_depth'] < 256:
    controller.submit_request(...)
```

### Address Alignment Errors

Ensure addresses are 8-byte aligned:

```python
addr = original_addr & ~0x7  # Align to 8-byte boundary
controller.submit_request(addr=addr, ...)
```

### Timing Violations

If `can_activate()` returns `False`, wait for timing constraints:

```python
while not bank.can_activate():
    channel.tick()
```

### DFI Not Ready

Wait for DFI to be ready before issuing commands:

```python
while not controller.dfi_ready:
    controller.tick()
```

## Examples

See the `examples/` directory for working examples:

| Example | Description |
|---------|-------------|
| `basic_controller.py` | Simple read/write operations |
| `multi_channel.py` | Multi-channel parallelism |
| `qos_scheduling.py` | Priority-based scheduling |
| `refresh_scheduling.py` | Refresh management |
| `bandwidth_benchmark.py` | Performance benchmarking |
| `dfi_interface.py` | DFI protocol usage |
| `address_decoding.py` | Address mapping examples |
| `dram_features.py` | DRAM timing and features |

Run any example:

```bash
python3 examples/basic_controller.py
python3 examples/multi_channel.py
python3 examples/qos_scheduling.py
```

## References

- JEDEC JESD270-4A HBM4 Specification
- Synopsys DesignWare HBM4/4E Controller IP
- CMU-SAFARI Ramulator2
- DFI 5.0/5.1 Specification