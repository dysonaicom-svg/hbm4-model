"""
HBM Exception Classes
参考设计文档 2026-06-15-hbm-system-model-design.md 的 7.3 节
"""


class HBMError(Exception):
    """HBM 协议错误基类
    
    所有 HBM 相关异常的基类。
    """
    pass


class AddressError(HBMError):
    """地址越界或对齐错误
    
    当访问地址超出有效范围或地址对齐不正确时抛出。
    """
    pass


class TimingError(HBMError):
    """时序违规
    
    当 DRAM 操作违反时序约束时抛出。
    例如：bank 未 ready 时发起 ACT。
    """
    pass


class QueueOverflowError(HBMError):
    """队列溢出
    
    当请求队列已满无法接受新请求时抛出。
    """
    pass


class ProtocolViolationError(HBMError):
    """协议违规
    
    当请求违反 HBM 协议规范时抛出。
    例如：无效的命令序列。
    """
    pass
