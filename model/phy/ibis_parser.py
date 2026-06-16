"""
IBIS Parser - Parse IBIS (I/O Buffer Information Specification) files

IBIS is a standard format for describing analog characteristics of IC I/O buffers.
This parser extracts model parameters, IV curves, and V-T waveforms from .ibs files.

Reference: IBIS (I/O Buffer Information Specification) Version 6.1

Key IBIS elements parsed:
- [Pullup] / [Pulldown]: Output driver IV characteristics
- [Rising Waveform] / [Falling Waveform]: V-T output waveforms
- [Composite Data Table]: Behavioral IBIS model data
- [Pin]: Package寄生参数
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re
import math


class IBISModelType(Enum):
    """IBIS model type enumeration"""
    INPUT = "Input"
    OUTPUT = "Output"
    IO = "I/O"  # I/O type (input/output)
    THREE_STATE = "3-state"
    OPEN_DRAIN = "Open_drain"
    TERMINATOR = "Terminator"


@dataclass
class IBISPackage:
    """Package RLC寄生参数"""
    r_pkg: float  # Resistance (ohms)
    l_pkg: float  # Inductance (nH)
    c_pkg: float  # Capacitance (pF)

    @classmethod
    def from_list(cls, values: List[float]) -> 'IBISPackage':
        """Parse from [R_pkg, L_pkg, C_pkg] list"""
        if len(values) < 3:
            raise ValueError(f"Package requires 3 values, got {len(values)}")
        return cls(r_pkg=values[0], l_pkg=values[1], c_pkg=values[2])


@dataclass
class IBISPin:
    """Pin definition with model mapping"""
    pin_name: str
    model_name: str
    r_pin: float = 0.0  # Pin resistance (ohms)
    l_pin: float = 0.0  # Pin inductance (nH)
    c_pin: float = 0.0  # Pin capacitance (pF)


@dataclass
class IVCurve:
    """IV characteristic curve"""
    voltage: List[float]  # Voltage points (V)
    current: List[float]  # Current points (A)

    def __post_init__(self):
        if len(self.voltage) != len(self.current):
            raise ValueError("IV curve must have equal voltage and current points")

    def interpolate(self, voltage: float) -> float:
        """Interpolate current at given voltage"""
        if len(self.voltage) < 2:
            return self.current[0] if self.current else 0.0

        # Find surrounding points
        for i in range(len(self.voltage) - 1):
            if self.voltage[i] <= voltage <= self.voltage[i + 1]:
                v1, v2 = self.voltage[i], self.voltage[i + 1]
                i1, i2 = self.current[i], self.current[i + 1]
                # Linear interpolation
                if v2 == v1:
                    return i1
                t = (voltage - v1) / (v2 - v1)
                return i1 + t * (i2 - i1)

        # Extrapolate based on end points
        if voltage < self.voltage[0]:
            return self.current[0]
        else:
            return self.current[-1]


@dataclass
class VTWaveform:
    """Voltage-Time output waveform"""
    time: List[float]  # Time points (ns)
    voltage: List[float]  # Voltage points (V)
    impedance: float  # Load impedance (ohms)
    v_com: float  # Common voltage (V)
    r_load: float  # Load resistance (ohms)

    def __post_init__(self):
        if len(self.time) != len(self.voltage):
            raise ValueError("VT waveform must have equal time and voltage points")

    def interpolate(self, time: float) -> float:
        """Interpolate voltage at given time"""
        if len(self.time) < 2:
            return self.voltage[0] if self.voltage else 0.0

        for i in range(len(self.time) - 1):
            if self.time[i] <= time <= self.time[i + 1]:
                t1, t2 = self.time[i], self.time[i + 1]
                v1, v2 = self.voltage[i], self.voltage[i + 1]
                if t2 == t1:
                    return v1
                t = (time - t1) / (t2 - t1)
                return v1 + t * (v2 - v1)

        if time < self.time[0]:
            return self.voltage[0]
        else:
            return self.voltage[-1]


@dataclass
class CompositeDataTable:
    """Composite Data Table (CDT) for behavioral IBIS models"""
    # Rising waveform data
    rising_time: List[float]
    rising_voltage: List[float]
    # Falling waveform data
    falling_time: List[float]
    falling_voltage: List[float]
    # Model parameters
    c_comp: float  # Comp node capacitance (pF)
    r_pin: float  # Pin resistance (ohms)
    l_pin: float  # Pin inductance (nH)
    v_cc: float  # Supply voltage (V)
    v_cpulldown: float  # Pull-down reference voltage
    v_cpullover: float  # Pull-up reference voltage


@dataclass
class IBISModel:
    """Complete IBIS model data"""
    model_name: str
    model_type: IBISModelType
    polarity: str = "Non-Inverting"
    enable: str = "Active"

    # IV characteristics
    pullup: Optional[IVCurve] = None
    pulldown: Optional[IVCurve] = None
    gnd_clamp: Optional[IVCurve] = None
    power_clamp: Optional[IVCurve] = None

    # V-T waveforms
    rising_waveform: Optional[VTWaveform] = None
    falling_waveform: Optional[VTWaveform] = None

    # Composite Data Table
    composite_data: Optional[CompositeDataTable] = None

    # Model parameters
    c_comp: float = 0.0  # Comp node capacitance (pF)
    c_comp_pulse: float = 0.0  # Pulse capacitance
    r_comp: float = 0.0  # Comp resistance (ohms)
    l_comp: float = 0.0  # Comp inductance (nH)

    # Reference voltages
    v_ref: float = 0.0  # Reference voltage for input models
    v_meas: float = 0.0  # Measurement voltage

    # Waveform parameters
    dV_dt_r: float = 0.0  # Rising edge dV/dt
    dV_dt_f: float = 0.0  # Falling edge dV/dt

    # Package parameters
    package: Optional[IBISPackage] = None

    # Manufacturer info
    manufacturer: str = ""
    product: str = ""

    def __post_init__(self):
        """Validate model consistency"""
        if self.model_type == IBISModelType.OUTPUT:
            if self.pullup is None and self.pulldown is None:
                pass  # Allow models without IV curves for testing


@dataclass
class IBISFile:
    """Complete IBIS file parsed data"""
    file_name: str = ""
    ibis_version: str = ""
    file_name_header: str = ""
    date: str = ""
    revision: str = ""

    # Component info
    component: str = ""
    manufacturer: str = ""

    # Package data
    default_package: Optional[IBISPackage] = None
    packages: Dict[str, IBISPackage] = field(default_factory=dict)

    # Pin mappings
    pins: Dict[str, IBISPin] = field(default_factory=dict)

    # Models
    models: Dict[str, IBISModel] = field(default_factory=dict)

    # Internal model references
    model_selectors: Dict[str, str] = field(default_factory=dict)


class IBISParser:
    """IBIS file parser

    Parses .ibs files according to IBIS specification.
    Handles all major IBIS sections including models, waveforms, and package data.
    """

    # Section markers - allow spaces in section names
    SECTION_PATTERN = re.compile(r'^\[([^\]]+)\]\s*$')
    # Key-value pattern - allow spaces in keys
    KEY_VALUE_PATTERN = re.compile(r'^([\w\s\-]+)\s*=\s*(.+)$')

    def __init__(self):
        self.line_number = 0

    def parse(self, content: str) -> IBISFile:
        """Parse IBIS file content

        Args:
            content: Raw IBIS file content as string

        Returns:
            IBISFile object with all parsed data
        """
        lines = content.split('\n')
        ibis_file = IBISFile()

        # Parse sections
        current_section = ""
        section_lines = []

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip purely blank lines at top level
            if not stripped:
                continue

            # Check for new section marker
            section_match = self.SECTION_PATTERN.match(stripped)
            if section_match:
                # Process previous section
                if current_section:
                    self._process_section(ibis_file, current_section, section_lines)

                current_section = section_match.group(1)
                section_lines = []
                self.line_number = line_num
            else:
                section_lines.append((line_num, stripped))

        # Process last section
        if current_section:
            self._process_section(ibis_file, current_section, section_lines)

        return ibis_file

    def parse_file(self, file_path: str) -> IBISFile:
        """Parse IBIS file from path

        Args:
            file_path: Path to .ibs file

        Returns:
            IBISFile object
        """
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return self.parse(content)

    def _is_comment_or_blank(self, line: str) -> bool:
        """Check if line is comment or blank"""
        if not line:
            return True
        if line.startswith('*'):
            return True
        return False

    def _process_section(self, ibis_file: IBISFile, section: str, lines: List[Tuple[int, str]]):
        """Process a single IBIS section"""
        handlers = {
            'File Header': self._parse_file_header,
            'Component': self._parse_component,
            'Package': self._parse_package,
            'Pin': self._parse_pin,
            'Model': self._parse_model,
            'Pullup': self._parse_pullup,
            'Pulldown': self._parse_pulldown,
            'GND Clamp': self._parse_gnd_clamp,
            'Power Clamp': self._parse_power_clamp,
            'Rising Waveform': self._parse_rising_waveform,
            'Falling Waveform': self._parse_falling_waveform,
            'Composite Data Table': self._parse_composite_data_table,
            'Submodel': self._parse_submodel,
            'Model Selector': self._parse_model_selector,
            'Input_Model': self._parse_input_model,
        }

        handler = handlers.get(section)
        if handler:
            handler(ibis_file, lines)
        # Unknown sections are silently ignored

    def _parse_file_header(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [File Header] section"""
        for _, line in lines:
            # Skip pure comments
            if line.startswith('*'):
                continue

            match = self.KEY_VALUE_PATTERN.match(line)
            if match:
                key, value = match.groups()
                key = key.strip()
                value = value.strip()
                if key == 'IBIS Version':
                    ibis_file.ibis_version = value
                elif key == 'File Name':
                    ibis_file.file_name_header = value
                elif key == 'Date':
                    ibis_file.date = value
                elif key == 'Revision':
                    ibis_file.revision = value

    def _parse_component(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [Component] section"""
        for _, line in lines:
            if line.startswith('*'):
                continue
            if not line:
                continue
            # Component name is entire first non-comment, non-blank line
            # May be just the name or "Name Manufacturer"
            ibis_file.component = line.strip()
            break

    def _parse_package(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [Package] section - handles both formats"""
        values = []

        for _, line in lines:
            if line.startswith('*'):
                continue
            if not line or line.startswith('|'):
                continue

            parts = line.split()
            if len(parts) >= 3:
                try:
                    # Check if values look like R L C (numbers)
                    if all(self._is_number(p) for p in parts[:3]):
                        r_val = float(parts[0])
                        l_val = float(parts[1])
                        c_val = float(parts[2])
                        ibis_file.default_package = IBISPackage(r_pkg=r_val, l_pkg=l_val, c_pkg=c_val)
                        return
                except ValueError:
                    pass

            # Check for key=value format like "R_pkg = 0.1"
            match = self.KEY_VALUE_PATTERN.match(line)
            if match:
                key, value = match.groups()
                key = key.strip()  # Strip trailing/leading spaces
                value = value.strip()
                try:
                    val = float(value)
                    values.append((key, val))
                except ValueError:
                    pass

        # If we got R_pkg, L_pkg, C_pkg from key=value format
        r_val = None
        l_val = None
        c_val = None
        for key, val in values:
            if key == 'R_pkg':
                r_val = val
            elif key == 'L_pkg':
                l_val = val
            elif key == 'C_pkg':
                c_val = val

        if r_val is not None and l_val is not None and c_val is not None:
            ibis_file.default_package = IBISPackage(r_pkg=r_val, l_pkg=l_val, c_pkg=c_val)

    def _is_number(self, s: str) -> bool:
        """Check if string is a number"""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _parse_pin(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [Pin] section - pin model mapping"""
        for _, line in lines:
            if line.startswith('*'):
                continue
            if not line or line.startswith('|'):
                continue

            parts = line.split()
            if len(parts) >= 2:
                pin_name = parts[0]
                model_name = parts[1]
                pin = IBISPin(pin_name=pin_name, model_name=model_name)

                # Optional RLC values (may be after model name)
                if len(parts) >= 5:
                    try:
                        pin.r_pin = float(parts[2])
                        pin.l_pin = float(parts[3])
                        pin.c_pin = float(parts[4])
                    except ValueError:
                        pass

                ibis_file.pins[pin_name] = pin

    def _parse_model(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [Model] section"""
        current_model_name = ""
        model_type = IBISModelType.OUTPUT
        model_params = {}

        for _, line in lines:
            if line.startswith('*'):
                continue
            if not line:
                continue

            # Strip leading | for key-value lines
            clean_line = line.lstrip('|').strip()

            # Check for model name (first line without =)
            if '=' not in clean_line:
                model_name = clean_line.strip()
                if model_name:
                    current_model_name = model_name
                    ibis_file.models[current_model_name] = IBISModel(
                        model_name=current_model_name,
                        model_type=IBISModelType.OUTPUT
                    )
                continue

            match = self.KEY_VALUE_PATTERN.match(clean_line)
            if match and current_model_name:
                key, value = match.groups()
                key = key.strip()
                value = value.strip()

                if key == 'Model_type':
                    try:
                        model_type = IBISModelType(value)
                    except ValueError:
                        model_type = IBISModelType.OUTPUT
                    model_params['model_type'] = model_type
                elif key == 'Polarity':
                    model_params['polarity'] = value
                elif key == 'Enable':
                    model_params['enable'] = value
                elif key == 'C_comp':
                    try:
                        model_params['c_comp'] = float(value)
                    except ValueError:
                        pass
                elif key == 'C_comp_pulse':
                    try:
                        model_params['c_comp_pulse'] = float(value)
                    except ValueError:
                        pass
                elif key == 'R_comp':
                    try:
                        model_params['r_comp'] = float(value)
                    except ValueError:
                        pass
                elif key == 'L_comp':
                    try:
                        model_params['l_comp'] = float(value)
                    except ValueError:
                        pass
                elif key == 'V_ref':
                    try:
                        model_params['v_ref'] = float(value)
                    except ValueError:
                        pass
                elif key == 'V_meas':
                    try:
                        model_params['v_meas'] = float(value)
                    except ValueError:
                        pass
                elif key == 'Manufacturer':
                    model_params['manufacturer'] = value
                elif key == 'Product':
                    model_params['product'] = value

        # Update model with parsed parameters
        if current_model_name and current_model_name in ibis_file.models:
            model = ibis_file.models[current_model_name]
            for key, value in model_params.items():
                setattr(model, key, value)

    def _parse_input_model(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [Input_Model] section (same as Model but for input types)"""
        self._parse_model(ibis_file, lines)

    def _parse_iv_curve(self, lines: List[Tuple[int, str]], expected_columns: int = 2) -> IVCurve:
        """Parse IV curve data table"""
        voltage = []
        current = []

        for _, line in lines:
            if line.startswith('*'):
                continue
            if not line or line.startswith('|'):
                continue

            parts = line.split()
            if len(parts) >= expected_columns:
                try:
                    # Handle negative values with V or I prefix
                    v_str = parts[0].strip()
                    i_str = parts[1].strip()

                    # Remove V/I prefixes if present (but keep - sign)
                    v_str = v_str.lstrip('VIvi')
                    i_str = i_str.lstrip('VIvi')

                    v = float(v_str)
                    i = float(i_str)
                    voltage.append(v)
                    current.append(i)
                except ValueError:
                    continue

        return IVCurve(voltage=voltage, current=current)

    def _parse_pullup(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [Pullup] section"""
        if not ibis_file.models:
            return

        curve = self._parse_iv_curve(lines)
        if curve.voltage:  # Only assign if we got data
            # Assign to most recently defined model
            model_names = list(ibis_file.models.keys())
            if model_names:
                ibis_file.models[model_names[-1]].pullup = curve

    def _parse_pulldown(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [Pulldown] section"""
        if not ibis_file.models:
            return

        curve = self._parse_iv_curve(lines)
        if curve.voltage:
            model_names = list(ibis_file.models.keys())
            if model_names:
                ibis_file.models[model_names[-1]].pulldown = curve

    def _parse_gnd_clamp(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [GND Clamp] section"""
        if not ibis_file.models:
            return

        curve = self._parse_iv_curve(lines)
        if curve.voltage:
            model_names = list(ibis_file.models.keys())
            if model_names:
                ibis_file.models[model_names[-1]].gnd_clamp = curve

    def _parse_power_clamp(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [Power Clamp] section"""
        if not ibis_file.models:
            return

        curve = self._parse_iv_curve(lines)
        if curve.voltage:
            model_names = list(ibis_file.models.keys())
            if model_names:
                ibis_file.models[model_names[-1]].power_clamp = curve

    def _parse_vt_waveform(self, lines: List[Tuple[int, str]]) -> VTWaveform:
        """Parse V-T waveform data"""
        time = []
        voltage = []
        impedance = 50.0  # Default
        v_com = 0.0
        r_load = 50.0

        for _, line in lines:
            if line.startswith('*'):
                continue
            if not line or line.startswith('|'):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            first = parts[0].strip()

            if first == 'R_load':
                try:
                    r_load = float(parts[1])
                except ValueError:
                    pass
            elif first == 'C_load':
                # Skip capacitance load (use R_load)
                pass
            elif first == 'V_com':
                try:
                    v_com = float(parts[1])
                except ValueError:
                    pass
            elif first == 'typ' or first == 'min' or first == 'max':
                # Waveform data line with typ/min/max prefix
                try:
                    t = float(parts[1])
                    v = float(parts[2])
                    time.append(t)
                    voltage.append(v)
                except ValueError:
                    continue
            else:
                # Data row: time voltage
                try:
                    t = float(first)
                    v = float(parts[1])
                    time.append(t)
                    voltage.append(v)
                except ValueError:
                    continue

        return VTWaveform(
            time=time,
            voltage=voltage,
            impedance=impedance,
            v_com=v_com,
            r_load=r_load
        )

    def _parse_rising_waveform(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [Rising Waveform] section"""
        if not ibis_file.models:
            return

        waveform = self._parse_vt_waveform(lines)
        if waveform.time:  # Only assign if we got data
            model_names = list(ibis_file.models.keys())
            if model_names:
                ibis_file.models[model_names[-1]].rising_waveform = waveform

    def _parse_falling_waveform(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [Falling Waveform] section"""
        if not ibis_file.models:
            return

        waveform = self._parse_vt_waveform(lines)
        if waveform.time:
            model_names = list(ibis_file.models.keys())
            if model_names:
                ibis_file.models[model_names[-1]].falling_waveform = waveform

    def _parse_composite_data_table(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [Composite Data Table] section"""
        if not ibis_file.models:
            return

        cdt = CompositeDataTable(
            rising_time=[],
            rising_voltage=[],
            falling_time=[],
            falling_voltage=[],
            c_comp=0.0,
            r_pin=0.0,
            l_pin=0.0,
            v_cc=0.0,
            v_cpulldown=0.0,
            v_cpullover=0.0
        )

        current_section = "rising"

        for _, line in lines:
            if line.startswith('*'):
                continue
            if not line or line.startswith('|'):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            first = parts[0].strip()

            # Parameter lines
            if first == 'C_comp':
                try:
                    cdt.c_comp = float(parts[1])
                except ValueError:
                    pass
            elif first == 'R_pin':
                try:
                    cdt.r_pin = float(parts[1])
                except ValueError:
                    pass
            elif first == 'L_pin':
                try:
                    cdt.l_pin = float(parts[1])
                except ValueError:
                    pass
            elif first == 'V_cc':
                try:
                    cdt.v_cc = float(parts[1])
                except ValueError:
                    pass
            elif first == 'V_cpulldown':
                try:
                    cdt.v_cpulldown = float(parts[1])
                except ValueError:
                    pass
            elif first == 'V_cpullover':
                try:
                    cdt.v_cpullover = float(parts[1])
                except ValueError:
                    pass
            elif first == '[Rising]':
                current_section = "rising"
            elif first == '[Falling]':
                current_section = "falling"
            else:
                # Data row
                try:
                    t = float(first)
                    v = float(parts[1])
                    if current_section == "rising":
                        cdt.rising_time.append(t)
                        cdt.rising_voltage.append(v)
                    else:
                        cdt.falling_time.append(t)
                        cdt.falling_voltage.append(v)
                except ValueError:
                    continue

        model_names = list(ibis_file.models.keys())
        if model_names:
            ibis_file.models[model_names[-1]].composite_data = cdt

    def _parse_submodel(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [Submodel] section - placeholder for now"""
        pass

    def _parse_model_selector(self, ibis_file: IBISFile, lines: List[Tuple[int, str]]):
        """Parse [Model Selector] section"""
        for _, line in lines:
            if line.startswith('*'):
                continue
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                selector_name = parts[0]
                model_ref = parts[1]
                ibis_file.model_selectors[selector_name] = model_ref


# Convenience functions
def parse_ibis_file(file_path: str) -> IBISFile:
    """Parse an IBIS file from path

    Args:
        file_path: Path to .ibs file

    Returns:
        IBISFile object with all parsed data
    """
    parser = IBISParser()
    return parser.parse_file(file_path)


def parse_ibis_content(content: str) -> IBISFile:
    """Parse IBIS content from string

    Args:
        content: Raw IBIS file content

    Returns:
        IBISFile object with all parsed data
    """
    parser = IBISParser()
    return parser.parse(content)