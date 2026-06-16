"""
HBM Controller Models

This package provides transaction-level models for the HBM memory controller:

Classes:
    - HBMController: Main controller model with scheduling and queue management
    - HBM4Controller: HBM4-specific controller with 32-channel support
    - AddressDecoder: Base address decoder
    - HBM4AddressDecoder: HBM4 address decoder with RBC/BCR/CRB mapping
    - QoSScheduler: QoS-aware request scheduler
    - HBM4QoSScheduler: HBM4 QoS scheduler with 16-level priority
    - RefreshScheduler: DRAM refresh scheduler
    - HBM4RefreshScheduler: HBM4 refresh with per-bank and DRFM modes
    - RequestQueue: Thread-safe request queue
    - HBMRequest: Memory request representation
    - HBMConfig: Controller configuration

Usage:
    from model.controller.hbm4_controller import HBM4Controller

    controller = HBM4Controller()
    controller.submit_request(addr=0x1000, is_read=True)
"""

from model.controller.controller import HBMController
from model.controller.hbm4_controller import HBM4Controller
from model.controller.hbm4_address_decoder import HBM4AddressDecoder
from model.controller.hbm4_qos_scheduler import HBM4QoSScheduler, QoSLevel
from model.controller.hbm4_refresh_scheduler import HBM4RefreshScheduler, RefreshMode
from model.controller.queue import RequestQueue, ReadQueue, WriteQueue, QueueManager
from model.controller.request import HBMRequest, HBMResponse, RequestState
from model.controller.config import HBMConfig, HBM3_DEFAULT, HBM4_DEFAULT
from model.controller.exceptions import HBMError, AddressError, TimingError

__all__ = [
    'HBMController',
    'HBM4Controller',
    'HBM4AddressDecoder',
    'HBM4QoSScheduler',
    'QoSLevel',
    'HBM4RefreshScheduler',
    'RefreshMode',
    'RequestQueue',
    'ReadQueue',
    'WriteQueue',
    'QueueManager',
    'HBMRequest',
    'HBMResponse',
    'RequestState',
    'HBMConfig',
    'HBM3_DEFAULT',
    'HBM4_DEFAULT',
    'HBMError',
    'AddressError',
    'TimingError',
]