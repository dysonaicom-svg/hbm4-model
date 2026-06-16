"""
Comprehensive Tests for DRAM Bank State Machine

Tests cover:
1. Basic state transitions (ACT, PRE, READ, WRITE, REFRESH)
2. Timing constraint verification (tRC, tRCD, tRAS, tRP, tCCD)
3. Row hit/miss detection
4. Turnaround timing (tRTW, tWTRS, tWTRL)
5. Power management (power down, self refresh)
6. Error handling and edge cases
7. HBM4 timing parameter compatibility
"""

import pytest
from model.dram.bank_state_machine import (
    BankStateMachine, BankStateEnum, Bank, OperationType,
    TimingViolation
)
from model.dram.timing import HBM3Timing, HBM4Timing


class TestBankBasic:
    """Test Bank basic functionality"""

    def test_bank_creation(self):
        """测试 Bank 创建"""
        bank = Bank(bank_id=0)
        assert bank.bank_id == 0
        assert bank.state == BankStateEnum.IDLE
        assert bank.open_row == -1

    def test_bank_idle_property(self):
        """测试 is_idle 属性"""
        bank = Bank(bank_id=0)
        assert bank.is_idle
        bank.state = BankStateEnum.ACTIVE
        assert not bank.is_idle

    def test_bank_active_property(self):
        """测试 is_active 属性"""
        bank = Bank(bank_id=0)
        assert not bank.is_active
        bank.state = BankStateEnum.ACTIVE
        assert bank.is_active

    def test_bank_busy_property(self):
        """测试 is_busy 属性"""
        bank = Bank(bank_id=0)
        assert not bank.is_busy
        bank.state = BankStateEnum.BUSY
        assert bank.is_busy

    def test_bank_refresh_property(self):
        """测试 is_refresh 属性"""
        bank = Bank(bank_id=0)
        assert not bank.is_refresh
        bank.state = BankStateEnum.REFRESHING
        assert bank.is_refresh

    def test_bank_row_open(self):
        """测试 row_open 属性"""
        bank = Bank(bank_id=0)
        assert not bank.row_open
        bank.state = BankStateEnum.ACTIVE
        bank.open_row = 100
        assert bank.row_open

    def test_has_been_activated(self):
        """测试 has_been_activated 属性"""
        bank = Bank(bank_id=0)
        assert not bank.has_been_activated
        bank.activate_time = 10.0
        assert bank.has_been_activated

    def test_has_been_precharged(self):
        """测试 has_been_precharged 属性"""
        bank = Bank(bank_id=0)
        assert not bank.has_been_precharged
        bank.precharge_time = 10.0
        assert bank.has_been_precharged


class TestBankStateMachineCreation:
    """Test BankStateMachine creation"""

    def test_creation_with_hbm3_timing(self):
        """测试 HBM3 时序创建"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        assert sm.bank.bank_id == 0
        assert sm.bank.state == BankStateEnum.IDLE
        assert sm.current_time == 0.0

    def test_creation_with_hbm4_timing(self):
        """测试 HBM4 时序创建"""
        timing = HBM4Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        assert sm.bank.bank_id == 0
        assert sm.bank.state == BankStateEnum.IDLE

    def test_set_time(self):
        """测试时间设置"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(100.5)
        assert sm.current_time == 100.5


class TestActivationTransitions:
    """Test activation state transitions"""

    def test_can_activate_idle_bank(self):
        """测试 IDLE 状态 bank 可以激活"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(0.0)
        assert sm.can_activate()

    def test_cannot_activate_active_bank(self):
        """测试 ACTIVE 状态 bank 不能再次激活"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(10.0)
        sm.activate(row=0x100)
        assert not sm.can_activate()

    def test_cannot_activate_busy_bank(self):
        """测试 BUSY 状态 bank 不能激活"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD + 1)
        sm.read()
        assert not sm.can_activate()

    def test_activate_idle_bank_success(self):
        """测试激活 IDLE bank 成功"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(10.0)

        success, msg = sm.activate(row=0x100)

        assert success
        assert msg is None
        assert sm.bank.state == BankStateEnum.ACTIVE
        assert sm.bank.open_row == 0x100
        assert sm.bank.activate_time == 10.0

    def test_activate_records_operation_time(self):
        """测试激活记录操作时间"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(50.0)
        sm.activate(row=0x100)
        assert sm.bank.last_operation_time == 50.0

    def test_activate_failure_wrong_state(self):
        """测试激活失败：状态不对"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(100.0)

        success, msg = sm.activate(row=0x200)

        assert not success
        assert msg is not None
        assert "not idle" in msg.lower()

    def test_activate_failure_tRC_violation(self):
        """测试激活失败：tRC 违规"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        # 第一次激活
        sm.set_time(0.0)
        sm.activate(row=0x100)

        # 预充电
        sm.set_time(timing.tRAS + 1)
        sm.precharge()

        # tRC 时间内再次激活应该失败
        sm.set_time(timing.tRC - 1)
        success, msg = sm.activate(row=0x200)

        assert not success
        assert msg is not None
        assert "tRC" in msg

    def test_activate_after_tRC_ok(self):
        """测试 tRC 后激活成功"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)

        sm.set_time(timing.tRAS + 1)
        sm.precharge()
        precharge_time = sm.current_time

        sm.set_time(precharge_time + timing.tRC)
        success, msg = sm.activate(row=0x200)

        assert success
        assert msg is None


class TestPrechargeTransitions:
    """Test precharge state transitions"""

    def test_cannot_precharge_idle_bank(self):
        """测试不能预充电 IDLE bank"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(0.0)
        assert not sm.can_precharge()

    def test_can_precharge_active_bank_after_tras(self):
        """测试 tRAS 后可以预充电"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRAS + 1)

        assert sm.can_precharge()

    def test_cannot_precharge_before_tras(self):
        """测试 tRAS 前不能预充电"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRAS - 1)

        assert not sm.can_precharge()

    def test_precharge_success(self):
        """测试预充电成功"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRAS + 1)

        success, msg = sm.precharge()

        assert success
        assert msg is None
        assert sm.bank.state == BankStateEnum.IDLE
        assert sm.bank.open_row == -1

    def test_precharge_failure_tRAS_violation(self):
        """测试预充电失败：tRAS 违规"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRAS - 1)

        success, msg = sm.precharge()

        assert not success
        assert msg is not None
        assert "tRAS" in msg


class TestReadTransitions:
    """Test read state transitions"""

    def test_cannot_read_idle_bank(self):
        """测试不能读 IDLE bank"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(0.0)
        assert not sm.can_read()

    def test_cannot_read_before_trcd(self):
        """测试 tRCD 前不能读"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD - 1)

        assert not sm.can_read()

    def test_can_read_after_trcd(self):
        """测试 tRCD 后可以读"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD)

        assert sm.can_read()

    def test_read_success(self):
        """测试读操作成功"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD)

        success, msg = sm.read()

        assert success
        assert msg is None
        assert sm.bank.state == BankStateEnum.BUSY
        assert sm.bank.read_start_time >= 0

    def test_read_failure_wrong_state(self):
        """测试读失败：状态不对"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(0.0)

        success, msg = sm.read()

        assert not success
        assert msg is not None

    def test_complete_read_success(self):
        """测试读完成成功"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD)
        sm.read()

        # 前进到读完成后
        sm.set_time(sm.bank.read_complete_time + 1)
        success, msg = sm.complete_read()

        assert success
        assert msg is None
        assert sm.bank.state == BankStateEnum.ACTIVE
        assert sm.bank.read_start_time < 0

    def test_can_complete_read(self):
        """测试 can_complete_read"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD)
        sm.read()

        assert not sm.can_complete_read()

        sm.set_time(sm.bank.read_complete_time + 1)
        assert sm.can_complete_read()


class TestWriteTransitions:
    """Test write state transitions"""

    def test_cannot_write_idle_bank(self):
        """测试不能写 IDLE bank"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(0.0)
        assert not sm.can_write()

    def test_cannot_write_before_trcd(self):
        """测试 tRCD 前不能写"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD - 1)

        assert not sm.can_write()

    def test_can_write_after_trcd(self):
        """测试 tRCD 后可以写"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD)

        assert sm.can_write()

    def test_write_success(self):
        """测试写操作成功"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD)

        success, msg = sm.write()

        assert success
        assert msg is None
        assert sm.bank.state == BankStateEnum.BUSY
        assert sm.bank.write_start_time >= 0

    def test_complete_write_success(self):
        """测试写完成成功"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD)
        sm.write()

        sm.set_time(sm.bank.write_complete_time + 1)
        success, msg = sm.complete_write()

        assert success
        assert msg is None
        assert sm.bank.state == BankStateEnum.ACTIVE


class TestRefreshTransitions:
    """Test refresh state transitions"""

    def test_can_refresh_idle_bank(self):
        """测试可以刷新 IDLE bank"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(0.0)
        # First refresh should always work (has_been_activated check removed)
        assert sm.can_refresh()

    def test_cannot_refresh_active_bank(self):
        """测试不能刷新 ACTIVE bank"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)

        assert not sm.can_refresh()

    def test_refresh_success(self):
        """测试刷新成功"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        success, msg = sm.refresh()

        assert success
        assert msg is None
        assert sm.bank.state == BankStateEnum.REFRESHING
        assert sm.bank.refresh_time == 0.0

    def test_refresh_sets_complete_time(self):
        """测试刷新设置完成时间"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(100.0)
        sm.refresh()

        assert sm.bank.refresh_complete_time == 100.0 + timing.tRFC

    def test_complete_refresh_success(self):
        """测试刷新完成成功"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.refresh()
        sm.set_time(timing.tRFC)

        success, msg = sm.complete_refresh()

        assert success
        assert msg is None
        assert sm.bank.state == BankStateEnum.IDLE

    def test_cannot_complete_refresh_before_trfc(self):
        """测试 tRFC 前不能完成刷新"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.refresh()
        sm.set_time(timing.tRFC - 1)

        success, msg = sm.complete_refresh()

        assert not success
        assert msg is not None


class TestPowerManagement:
    """Test power management states"""

    def test_can_enter_power_down_from_idle(self):
        """测试可以从 IDLE 进入掉电"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(0.0)
        assert sm.can_enter_power_down()

    def test_cannot_enter_power_down_from_active(self):
        """测试不能从 ACTIVE 进入掉电"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)

        assert not sm.can_enter_power_down()

    def test_enter_power_down_success(self):
        """测试进入掉电成功"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        success, msg = sm.enter_power_down()

        assert success
        assert msg is None
        assert sm.bank.state == BankStateEnum.POWERDN

    def test_exit_power_down_success(self):
        """测试退出掉电成功"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.enter_power_down()
        success, msg = sm.exit_power_down()

        assert success
        assert msg is None
        assert sm.bank.state == BankStateEnum.IDLE

    def test_can_enter_self_refresh_from_idle(self):
        """测试可以从 IDLE 进入自刷新"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(0.0)
        assert sm.can_enter_self_refresh()

    def test_enter_self_refresh_success(self):
        """测试进入自刷新成功"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        success, msg = sm.enter_self_refresh()

        assert success
        assert msg is None
        assert sm.bank.state == BankStateEnum.SELFREF

    def test_exit_self_refresh_success(self):
        """测试退出自刷新成功"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.enter_self_refresh()
        success, msg = sm.exit_self_refresh()

        assert success
        assert msg is None
        assert sm.bank.state == BankStateEnum.IDLE


class TestRowHitMiss:
    """Test row hit/miss detection"""

    def test_is_row_hit(self):
        """测试 row hit 检测"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)

        assert sm.is_row_hit(0x100)
        assert not sm.is_row_hit(0x200)

    def test_is_row_open(self):
        """测试 is_row_open"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)

        assert sm.is_row_open(0x100)
        assert not sm.is_row_open(0x200)

    def test_close_row(self):
        """测试 close_row"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRAS + 1)

        success, msg = sm.close_row()

        assert success
        assert sm.bank.state == BankStateEnum.IDLE
        assert sm.bank.open_row == -1


class TestTurnaroundTiming:
    """Test turnaround timing (tRTW, tWTRS, tWTRL)"""

    def test_can_read_after_write_same_row(self):
        """测试同一行写后读"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD)
        sm.write()
        sm.set_time(sm.bank.write_complete_time + 1)
        sm.complete_write()

        # tWTRS 后可以读
        sm.set_time(sm.bank.write_complete_time + timing.nWTRS)
        assert sm.can_read_after_write()

    def test_can_write_after_read_same_row(self):
        """测试同一行读后写"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD)
        sm.read()
        sm.set_time(sm.bank.read_complete_time + 1)
        sm.complete_read()

        # tRTW 后可以写
        sm.set_time(sm.bank.read_complete_time + timing.nRTW)
        assert sm.can_write_after_read()


class TestTimingQuery:
    """Test timing query methods"""

    def test_time_to_ready_idle_bank(self):
        """测试 IDLE bank 立即 ready"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(0.0)
        assert sm.time_to_ready() == 0.0

    def test_time_to_ready_active_bank(self):
        """测试 ACTIVE bank 不可用"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)

        assert sm.time_to_ready() == float('inf')

    def test_time_to_read_ready(self):
        """测试 time_to_read_ready"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(10.0)

        # 需要等待 tRCD - 10
        remaining = sm.time_to_read_ready()
        assert 0 < remaining <= timing.tRCD - 10 + 1  # 允许一些误差

    def test_time_to_precharge_ready(self):
        """测试 time_to_precharge_ready"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(20.0)

        # 需要等待 tRAS - 20
        remaining = sm.time_to_precharge_ready()
        assert 0 < remaining <= timing.tRAS - 20 + 1


class TestViolations:
    """Test timing violation tracking"""

    def test_violations_recorded(self):
        """测试违规被记录"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRAS - 1)
        sm.precharge()

        violations = sm.get_violations()
        assert len(violations) > 0
        assert any(v.violation_type == 'tRAS' for v in violations)

    def test_clear_violations(self):
        """测试清除违规"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRAS - 1)
        sm.precharge()

        sm.clear_violations()
        assert len(sm.get_violations()) == 0


class TestHBM4Compatibility:
    """Test HBM4 timing compatibility"""

    def test_hbm4_timing_parameters(self):
        """测试 HBM4 时序参数"""
        timing = HBM4Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        # 验证可以获取 n-prefix 参数
        assert sm.get_timing_value('nRCD') == timing.nRCD
        assert sm.get_timing_value('nRAS') == timing.nRAS
        assert sm.get_timing_value('nRC') == timing.nRC
        assert sm.get_timing_value('nCCD') == timing.nCCD

    def test_hbm4_activation_cycle(self):
        """测试 HBM4 激活周期"""
        timing = HBM4Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        # 第一次激活（无 tRC 限制）
        sm.set_time(0.0)
        success, _ = sm.activate(row=0x100)
        assert success

        # 预充电 (after tRAS)
        sm.set_time(timing.nRAS + 1)
        sm.precharge()
        precharge_time = sm.current_time

        # 第二次激活需要满足 tRC
        sm.set_time(precharge_time + timing.nRC)
        success, _ = sm.activate(row=0x200)
        assert success


class TestMultipleBanks:
    """Test multiple banks independence"""

    def test_independent_banks(self):
        """测试多 bank 独立性"""
        timing = HBM3Timing()
        sm0 = BankStateMachine(bank_id=0, timing=timing)
        sm1 = BankStateMachine(bank_id=1, timing=timing)

        sm0.set_time(0.0)
        sm0.activate(row=0x100)

        sm1.set_time(0.0)
        sm1.activate(row=0x200)

        assert sm0.bank.is_active
        assert sm1.bank.is_active
        assert sm0.bank.open_row == 0x100
        assert sm1.bank.open_row == 0x200

    def test_bank0_busy_bank1_independent(self):
        """测试 bank0 BUSY 时 bank1 独立操作"""
        timing = HBM3Timing()
        sm0 = BankStateMachine(bank_id=0, timing=timing)
        sm1 = BankStateMachine(bank_id=1, timing=timing)

        # Bank0 进行读操作
        sm0.set_time(0.0)
        sm0.activate(row=0x100)
        sm0.set_time(timing.tRCD)
        sm0.read()

        # Bank1 独立激活
        sm1.set_time(0.0)
        sm1.activate(row=0x200)

        assert sm0.bank.is_busy
        assert sm1.bank.is_active


class TestOperationCompletion:
    """Test operation completion helpers"""

    def test_is_operation_in_progress(self):
        """测试 is_operation_in_progress"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        assert not sm.is_operation_in_progress()

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD)
        sm.read()

        assert sm.is_operation_in_progress()

    def test_read_write_mutual_exclusion(self):
        """测试读写互斥"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD)
        sm.read()

        # 读期间不能写
        assert not sm.can_write()


class TestComplexSequences:
    """Test complex command sequences"""

    def test_full_read_sequence(self):
        """测试完整读序列: ACT -> READ -> complete -> PRE"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        # ACT
        sm.set_time(0.0)
        sm.activate(row=0x100)

        # READ
        sm.set_time(timing.tRCD)
        sm.read()
        read_complete = sm.bank.read_complete_time

        # complete READ
        sm.set_time(read_complete + 1)
        sm.complete_read()
        complete_time = sm.current_time

        # PRE (after tRAS is satisfied)
        sm.set_time(timing.tRAS + 1)
        success, _ = sm.precharge()

        assert success
        assert sm.bank.state == BankStateEnum.IDLE

    def test_full_write_sequence(self):
        """测试完整写序列: ACT -> WRITE -> complete -> PRE"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        # ACT
        sm.set_time(0.0)
        sm.activate(row=0x100)

        # WRITE
        sm.set_time(timing.tRCD)
        sm.write()
        write_complete = sm.bank.write_complete_time

        # complete WRITE
        sm.set_time(write_complete + 1)
        sm.complete_write()

        # PRE (after tRAS is satisfied)
        sm.set_time(timing.tRAS + 1)
        success, _ = sm.precharge()

        assert success
        assert sm.bank.state == BankStateEnum.IDLE

    def test_read_write_turnaround(self):
        """测试读写 turnaround"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        # ACT
        sm.set_time(0.0)
        sm.activate(row=0x100)

        # READ
        sm.set_time(timing.tRCD)
        sm.read()
        read_complete = sm.bank.read_complete_time

        # complete READ
        sm.set_time(read_complete + 1)
        sm.complete_read()

        # 等待 tRTW
        sm.set_time(read_complete + timing.nRTW + 1)

        # WRITE
        assert sm.can_write()
        sm.write()

    def test_write_read_turnaround(self):
        """测试写读 turnaround"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        # ACT
        sm.set_time(0.0)
        sm.activate(row=0x100)

        # WRITE
        sm.set_time(timing.tRCD)
        sm.write()
        write_complete = sm.bank.write_complete_time

        # complete WRITE
        sm.set_time(write_complete + 1)
        sm.complete_write()

        # 等待 tWTRS
        sm.set_time(write_complete + timing.nWTRS + 1)

        # READ
        assert sm.can_read()
        sm.read()


class TestEdgeCases:
    """Test edge cases"""

    def test_activate_with_zero_time(self):
        """测试在 time=0 激活"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(0.0)

        success, _ = sm.activate(row=0x100)
        assert success

    def test_multiple_precharge_calls(self):
        """测试多次预充电调用"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRAS + 1)
        sm.precharge()

        # 第二次预充电应该失败
        success, msg = sm.precharge()
        assert not success

    def test_read_without_activation(self):
        """测试无激活读"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)
        sm.set_time(0.0)

        success, msg = sm.read()
        assert not success

    def test_power_down_after_refresh(self):
        """测试刷新后掉电"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.refresh()
        sm.set_time(timing.tRFC)
        sm.complete_refresh()

        sm.set_time(timing.tRFC + 1)
        success, _ = sm.enter_power_down()

        assert success

    def test_self_refresh_after_refresh(self):
        """测试刷新后自刷新"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.refresh()
        sm.set_time(timing.tRFC)
        sm.complete_refresh()

        sm.set_time(timing.tRFC + 1)
        success, _ = sm.enter_self_refresh()

        assert success


class TestLegacyCompatibility:
    """Test legacy method compatibility"""

    def test_complete_read_legacy(self):
        """测试 legacy complete_read"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD)
        sm.read()
        sm.set_time(sm.bank.read_complete_time + 1)

        sm.complete_read_legacy()
        assert sm.bank.state == BankStateEnum.ACTIVE

    def test_complete_write_legacy(self):
        """测试 legacy complete_write"""
        timing = HBM3Timing()
        sm = BankStateMachine(bank_id=0, timing=timing)

        sm.set_time(0.0)
        sm.activate(row=0x100)
        sm.set_time(timing.tRCD)
        sm.write()
        sm.set_time(sm.bank.write_complete_time + 1)

        sm.complete_write_legacy()
        assert sm.bank.state == BankStateEnum.ACTIVE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
