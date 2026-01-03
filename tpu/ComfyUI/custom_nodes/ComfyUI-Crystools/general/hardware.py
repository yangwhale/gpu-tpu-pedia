import platform
import re
import cpuinfo
from cpuinfo import DataSource
import psutil
from .hdd import getDrivesInfo

from ..core import logger

# Auto-detect TPU and use appropriate info class
def _get_device_info_class():
    """Returns CGPUInfo from tpu.py if TPU is detected, otherwise from gpu.py"""
    try:
        # Try to detect TPU via tpu_info
        from tpu_info import device as tpu_device
        chips_info = tpu_device.get_local_chips()
        if chips_info and chips_info[1] > 0:
            from .tpu import CGPUInfo
            logger.info("Using TPU monitoring (tpu.py)")
            return CGPUInfo
    except ImportError:
        pass
    except Exception:
        pass
    
    # Try JAX TPU detection
    try:
        import jax
        devices = jax.devices()
        for device in devices:
            if 'tpu' in str(device).lower():
                from .tpu import CGPUInfo
                logger.info("Using TPU monitoring (tpu.py)")
                return CGPUInfo
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback to GPU
    from .gpu import CGPUInfo
    return CGPUInfo

CGPUInfo = _get_device_info_class()


class CHardwareInfo:
    """
    This is only class to get information from hardware.
    Specially for share it to other software.
    """
    switchCPU = False
    switchHDD = False
    switchRAM = False
    whichHDD = '/' # breaks linux

    @property
    def switchGPU(self):
        return self.GPUInfo.switchGPU
    @switchGPU.setter
    def switchGPU(self, value):
        self.GPUInfo.switchGPU = value

    @property
    def switchVRAM(self):
        return self.GPUInfo.switchVRAM
    @switchVRAM.setter
    def switchVRAM(self, value):
        self.GPUInfo.switchVRAM = value

    def __init__(self, switchCPU=False, switchGPU=False, switchHDD=False, switchRAM=False, switchVRAM=False):
        self.switchCPU = switchCPU
        self.switchHDD = switchHDD
        self.switchRAM = switchRAM

        self.print_sys_info()

        self.GPUInfo = CGPUInfo()
        self.switchGPU = switchGPU
        self.switchVRAM = switchVRAM

    def print_sys_info(self):
        brand = None
        if DataSource.is_windows:   # Windows
            brand = DataSource.winreg_processor_brand().strip()
        elif DataSource.has_proc_cpuinfo():   # Linux
            return_code, output = DataSource.cat_proc_cpuinfo()
            if return_code == 0 and output is not None:
                for line in output.splitlines():
                    r = re.search(r'model name\s*:\s*(.+)', line)
                    if r:
                        brand = r.group(1)
                        break
        elif DataSource.has_sysctl():   # macOS
            return_code, output = DataSource.sysctl_machdep_cpu_hw_cpufrequency()
            if return_code == 0 and output is not None:
                for line in output.splitlines():
                    r = re.search(r'machdep\.cpu\.brand_string\s*:\s*(.+)', line)
                    if r:
                        brand = r.group(1)
                        break

        # fallback to use cpuinfo.get_cpu_info()
        if not brand:
            brand = cpuinfo.get_cpu_info().get('brand_raw', "Unknown")

        arch_string_raw = 'Arch unknown'

        try:
            arch_string_raw = DataSource.arch_string_raw
        except:
            pass

        specName = 'CPU: ' + brand
        specArch = 'Arch: ' + arch_string_raw
        specOs = 'OS: ' + str(platform.system()) + ' ' + str(platform.release())
        logger.info(f"{specName} - {specArch} - {specOs}")

    def getHDDsInfo(self):
        return getDrivesInfo()

    def getGPUInfo(self):
        return self.GPUInfo.getInfo()

    def getStatus(self):
        cpu = -1
        ramTotal = -1
        ramUsed = -1
        ramUsedPercent = -1
        hddTotal = -1
        hddUsed = -1
        hddUsedPercent = -1

        if self.switchCPU:
            cpu = psutil.cpu_percent()

        if self.switchRAM:
            ram = psutil.virtual_memory()
            ramTotal = ram.total
            ramUsed = ram.used
            ramUsedPercent = ram.percent

        if self.switchHDD:
            try:
                hdd = psutil.disk_usage(self.whichHDD)
                hddTotal = hdd.total
                hddUsed = hdd.used
                hddUsedPercent = hdd.percent
            except Exception as e:
                logger.error(f"Error getting disk usage for {self.whichHDD}: {e}")
                hddTotal = -1
                hddUsed = -1
                hddUsedPercent = -1

        getStatus = self.GPUInfo.getStatus()

        return {
            'cpu_utilization': cpu,
            'ram_total': ramTotal,
            'ram_used': ramUsed,
            'ram_used_percent': ramUsedPercent,
            'hdd_total': hddTotal,
            'hdd_used': hddUsed,
            'hdd_used_percent': hddUsedPercent,
            'device_type': getStatus['device_type'],
            'gpus': getStatus['gpus'],
        }
