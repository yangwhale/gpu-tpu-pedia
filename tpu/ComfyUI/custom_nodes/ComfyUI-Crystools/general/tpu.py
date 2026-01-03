import torch
import comfy.model_management
from ..core import logger
import os
import platform

def is_tpu() -> bool:
    """
    Determines if a TPU is available by checking tpu_info or JAX devices.
    """
    # Try tpu_info first (more reliable for system detection)
    try:
        from tpu_info import device as tpu_device
        chips_info = tpu_device.get_local_chips()
        if chips_info and chips_info[1] > 0:
            chip_type = chips_info[0].value.name
            chip_count = chips_info[1]
            hbm_gib = chips_info[0].value.hbm_gib
            logger.info(f"TPU detected via tpu_info: {chip_count}x TPU {chip_type} ({hbm_gib} GiB HBM each)")
            return True
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"tpu_info detection failed: {e}")
    
    # Fallback to JAX detection
    try:
        import jax
        devices = jax.devices()
        for device in devices:
            if 'tpu' in str(device).lower():
                logger.info(f"TPU detected via JAX: {device}")
                return True
        return False
    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"JAX TPU detection failed: {e}")
        return False

IS_TPU = is_tpu()

def is_jetson() -> bool:
    """
    Determines if the Python environment is running on a Jetson device by checking the device model
    information or the platform release.
    """
    PROC_DEVICE_MODEL = ''
    try:
        with open('/proc/device-tree/model', 'r') as f:
            PROC_DEVICE_MODEL = f.read().strip()
            logger.info(f"Device model: {PROC_DEVICE_MODEL}")
            return "NVIDIA" in PROC_DEVICE_MODEL
    except Exception as e:
        # logger.warning(f"JETSON: Could not read /proc/device-tree/model: {e} (If you're not using Jetson, ignore this warning)")
        # If /proc/device-tree/model is not available, check platform.release()
        platform_release = platform.release()
        logger.info(f"Platform release: {platform_release}")
        if 'tegra' in platform_release.lower():
            logger.info("Detected 'tegra' in platform release. Assuming Jetson device.")
            return True
        else:
            logger.info("JETSON: Not detected.")
            return False

IS_JETSON = is_jetson()

class CGPUInfo:
    """
    This class is responsible for getting information from GPU/TPU.
    """
    cuda = False
    pynvmlLoaded = False
    jtopLoaded = False
    tpuLoaded = False
    tpuInfoLoaded = False  # tpu_info library loaded
    cudaAvailable = False
    torchDevice = 'cpu'
    cudaDevice = 'cpu'
    cudaDevicesFound = 0
    switchGPU = True
    switchVRAM = True
    switchTemperature = True
    gpus = []
    gpusUtilization = []
    gpusVRAM = []
    gpusTemperature = []
    
    # TPU specific
    jax = None
    tpuDevices = []
    tpuChipType = None  # tpu_info.device.TpuChip enum
    tpuChipTypeEnum = None  # Raw enum for get_chip_usage
    tpuChipCount = 0
    tpuHbmGib = 0
    
    # TPU metrics switches
    switchDutyCycle = True
    switchTensorCore = True
    gpusDutyCycle = []
    gpusTensorCore = []
    
    # Cached TPU usage data (updated periodically)
    _tpu_usage_cache = None
    _tpu_usage_cache_time = 0
    _tpu_tc_cache = None
    _tpu_tc_cache_time = 0

    def __init__(self):
        # Initialize instance attributes (override class defaults)
        self.gpus = []
        self.gpusUtilization = []
        self.gpusVRAM = []
        self.gpusTemperature = []
        self.tpuDevices = []
        self.gpusDutyCycle = []
        self.gpusTensorCore = []
        self._tpu_usage_cache = None
        self._tpu_usage_cache_time = 0
        self._tpu_tc_cache = None
        self._tpu_tc_cache_time = 0
        self.tpuChipTypeEnum = None
        self.tpuInfoLoaded = False
        self.tpuLoaded = False
        self.pynvmlLoaded = False
        self.jtopLoaded = False
        
        if IS_TPU:
            # Try to get TPU info from tpu_info library first
            try:
                from tpu_info import device as tpu_device
                chips_info = tpu_device.get_local_chips()
                if chips_info and chips_info[1] > 0:
                    self.tpuChipTypeEnum = chips_info[0]  # Store raw enum for get_chip_usage
                    self.tpuChipType = chips_info[0].value.name
                    self.tpuChipCount = chips_info[1]
                    self.tpuHbmGib = chips_info[0].value.hbm_gib
                    self.tpuInfoLoaded = True
                    logger.info(f'tpu_info initialized: {self.tpuChipCount}x TPU {self.tpuChipType}')
            except Exception as e:
                logger.debug(f'Could not get tpu_info: {e}')
            
            # Try to initialize TPU monitoring via JAX
            try:
                import jax
                self.jax = jax
                self.tpuDevices = [d for d in jax.devices() if 'tpu' in str(d).lower()]
                if self.tpuDevices:
                    self.tpuLoaded = True
                    logger.info(f'JAX TPU initialized. Found {len(self.tpuDevices)} TPU device(s).')
                    if self.tpuInfoLoaded:
                        logger.info(f'  TPU type: {self.tpuChipType}, {self.tpuChipCount} chips, {self.tpuHbmGib} GiB HBM each')
                    for i, dev in enumerate(self.tpuDevices):
                        logger.info(f"  TPU {i}: {dev}")
            except ImportError as e:
                logger.warning('JAX is not installed. ' + str(e))
            except Exception as e:
                # JAX TPU init failed, but we can still use tpu_info for monitoring
                logger.warning(f'JAX TPU init failed: {e}')
                if self.tpuInfoLoaded:
                    # Use tpu_info only mode - create fake device list
                    self.tpuLoaded = True  # Enable TPU monitoring using tpu_info
                    for i in range(self.tpuChipCount):
                        self.tpuDevices.append(None)  # Placeholder, we use tpu_info instead
                    logger.info(f'Using tpu_info for TPU monitoring ({self.tpuChipCount} devices)')
        elif IS_JETSON:
            # Try to import jtop for Jetson devices
            try:
                from jtop import jtop
                self.jtopInstance = jtop()
                self.jtopInstance.start()
                self.jtopLoaded = True
                logger.info('jtop initialized on Jetson device.')
            except ImportError as e:
                logger.error('jtop is not installed. ' + str(e))
            except Exception as e:
                logger.error('Could not initialize jtop. ' + str(e))
        else:
            # Try to import pynvml for non-Jetson devices
            try:
                import pynvml
                self.pynvml = pynvml
                self.pynvml.nvmlInit()
                self.pynvmlLoaded = True
                logger.info('pynvml (NVIDIA) initialized.')
            except ImportError as e:
                logger.error('pynvml is not installed. ' + str(e))
            except Exception as e:
                logger.error('Could not init pynvml (NVIDIA). ' + str(e))

        self.anygpuLoaded = self.pynvmlLoaded or self.jtopLoaded or self.tpuLoaded

        try:
            self.torchDevice = comfy.model_management.get_torch_device_name(comfy.model_management.get_torch_device())
        except Exception as e:
            logger.error('Could not pick default device. ' + str(e))

        if self.pynvmlLoaded and not self.jtopLoaded and not self.deviceGetCount():
            logger.warning('No GPU detected, disabling GPU monitoring.')
            self.anygpuLoaded = False
            self.pynvmlLoaded = False
            self.jtopLoaded = False

        if self.anygpuLoaded:
            if self.deviceGetCount() > 0:
                self.cudaDevicesFound = self.deviceGetCount()

                if self.tpuLoaded:
                    logger.info(f"TPU/s:")
                else:
                    logger.info(f"GPU/s:")

                for deviceIndex in range(self.cudaDevicesFound):
                    deviceHandle = self.deviceGetHandleByIndex(deviceIndex)

                    gpuName = self.deviceGetName(deviceHandle, deviceIndex)

                    logger.info(f"{deviceIndex}) {gpuName}")

                    self.gpus.append({
                        'index': deviceIndex,
                        'name': gpuName,
                    })

                    # Same index as gpus, with default values
                    self.gpusUtilization.append(True)
                    self.gpusVRAM.append(True)
                    self.gpusTemperature.append(True)
                    
                    # TPU specific metrics
                    if self.tpuLoaded:
                        self.gpusDutyCycle.append(True)
                        self.gpusTensorCore.append(True)

                self.cuda = True
                logger.info(self.systemGetDriverVersion())
            else:
                if self.tpuLoaded:
                    logger.warning('No TPU detected.')
                else:
                    logger.warning('No GPU with CUDA detected.')
        else:
            logger.warning('No GPU/TPU monitoring libraries available.')

        # Set device type
        if self.tpuLoaded:
            self.cudaDevice = 'tpu'
        elif self.torchDevice == 'cpu':
            self.cudaDevice = 'cpu'
        else:
            self.cudaDevice = 'cuda'
        self.cudaAvailable = torch.cuda.is_available()

        if self.cuda and self.cudaAvailable and self.torchDevice == 'cpu' and not self.tpuLoaded:
            logger.warning('CUDA is available, but torch is using CPU.')

    def getInfo(self):
        logger.debug('Getting GPUs info...')
        return self.gpus

    def getStatus(self):
        gpuUtilization = -1
        gpuTemperature = -1
        vramUsed = -1
        vramTotal = -1
        vramPercent = -1

        gpuType = ''
        gpus = []

        if self.cudaDevice == 'cpu':
            gpuType = 'cpu'
            gpus.append({
                'gpu_utilization': -1,
                'gpu_temperature': -1,
                'vram_total': -1,
                'vram_used': -1,
                'vram_used_percent': -1,
            })
        elif self.cudaDevice == 'tpu':
            gpuType = 'tpu'
            # TPU monitoring
            if self.tpuLoaded:
                # Refresh tpu_info cache for all TPU metrics
                usage_cache = self._refreshTPUUsageCache()
                
                for deviceIndex in range(len(self.tpuDevices)):
                    deviceHandle = self.deviceGetHandleByIndex(deviceIndex)

                    gpuUtilization = -1  # TPU doesn't expose utilization
                    gpuTemperature = -1  # TPU doesn't expose temperature
                    vramPercent = -1
                    vramUsed = -1
                    vramTotal = -1
                    dutyCycle = -1
                    tensorCoreUtil = -1

                    if self.switchVRAM and deviceIndex < len(self.gpusVRAM) and self.gpusVRAM[deviceIndex]:
                        try:
                            # Use tpu_info usage cache for accurate HBM usage (preferred)
                            if usage_cache and deviceIndex < len(usage_cache):
                                usage = usage_cache[deviceIndex]
                                vramUsed = usage.memory_usage
                                vramTotal = usage.total_memory
                            # Fallback to JAX device memory_stats (less accurate, shows JAX allocator only)
                            elif deviceHandle is not None:
                                memory = self.deviceGetMemoryInfo(deviceHandle)
                                vramUsed = memory['used']
                                vramTotal = memory['total']
                            
                            if vramTotal and vramTotal != 0:
                                vramPercent = vramUsed / vramTotal * 100
                        except Exception as e:
                            logger.debug('Could not get TPU memory info. ' + str(e))

                    # Get TPU-specific metrics
                    if self.switchDutyCycle and deviceIndex < len(self.gpusDutyCycle) and self.gpusDutyCycle[deviceIndex]:
                        dutyCycle = self.deviceGetDutyCycle(deviceHandle, deviceIndex)
                    
                    if self.switchTensorCore and deviceIndex < len(self.gpusTensorCore) and self.gpusTensorCore[deviceIndex]:
                        tensorCoreUtil = self.deviceGetTensorCoreUtilization(deviceHandle, deviceIndex)

                    gpus.append({
                        'gpu_utilization': gpuUtilization,
                        'gpu_temperature': gpuTemperature,
                        'vram_total': vramTotal,
                        'vram_used': vramUsed,
                        'vram_used_percent': vramPercent,
                        'duty_cycle': dutyCycle,
                        'tensorcore_util': tensorCoreUtil,
                    })
        else:
            gpuType = self.cudaDevice

            if self.anygpuLoaded and self.cuda and self.cudaAvailable:
                for deviceIndex in range(self.cudaDevicesFound):
                    deviceHandle = self.deviceGetHandleByIndex(deviceIndex)

                    gpuUtilization = -1
                    vramPercent = -1
                    vramUsed = -1
                    vramTotal = -1
                    gpuTemperature = -1

                    # GPU Utilization
                    if self.switchGPU and self.gpusUtilization[deviceIndex]:
                        try:
                            gpuUtilization = self.deviceGetUtilizationRates(deviceHandle)
                        except Exception as e:
                            logger.error('Could not get GPU utilization. ' + str(e))
                            logger.error('Monitor of GPU is turning off.')
                            self.switchGPU = False

                    if self.switchVRAM and self.gpusVRAM[deviceIndex]:
                        try:
                            memory = self.deviceGetMemoryInfo(deviceHandle)
                            vramUsed = memory['used']
                            vramTotal = memory['total']

                            # Check if vramTotal is not zero or None
                            if vramTotal and vramTotal != 0:
                                vramPercent = vramUsed / vramTotal * 100
                        except Exception as e:
                            logger.error('Could not get GPU memory info. ' + str(e))
                            self.switchVRAM = False

                    # Temperature
                    if self.switchTemperature and self.gpusTemperature[deviceIndex]:
                        try:
                            gpuTemperature = self.deviceGetTemperature(deviceHandle)
                        except Exception as e:
                            logger.error('Could not get GPU temperature. Turning off this feature. ' + str(e))
                            self.switchTemperature = False

                    gpus.append({
                        'gpu_utilization': gpuUtilization,
                        'gpu_temperature': gpuTemperature,
                        'vram_total': vramTotal,
                        'vram_used': vramUsed,
                        'vram_used_percent': vramPercent,
                    })

        return {
            'device_type': gpuType,
            'gpus': gpus,
        }

    def deviceGetCount(self):
        if self.tpuLoaded:
            return len(self.tpuDevices)
        elif self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetCount()
        elif self.jtopLoaded:
            # For Jetson devices, we assume there's one GPU
            return 1
        else:
            return 0

    def deviceGetHandleByIndex(self, index):
        if self.tpuLoaded:
            return self.tpuDevices[index] if index < len(self.tpuDevices) else None
        elif self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetHandleByIndex(index)
        elif self.jtopLoaded:
            return index  # On Jetson, index acts as handle
        else:
            return 0

    def deviceGetName(self, deviceHandle, deviceIndex):
        if self.tpuLoaded:
            try:
                # deviceHandle is the JAX device object (or None if JAX failed)
                if deviceHandle is not None:
                    return str(deviceHandle)
                else:
                    # When using tpu_info only mode, create a simple name
                    return f'TPU {deviceIndex} ({self.tpuChipType})'
            except Exception as e:
                logger.error('Could not get TPU name. ' + str(e))
                return f'TPU {deviceIndex}'
        elif self.pynvmlLoaded:
            gpuName = 'Unknown GPU'

            try:
                gpuName = self.pynvml.nvmlDeviceGetName(deviceHandle)
                try:
                    gpuName = gpuName.decode('utf-8', errors='ignore')
                except AttributeError:
                    pass

            except UnicodeDecodeError as e:
                gpuName = 'Unknown GPU (decoding error)'
                logger.error(f"UnicodeDecodeError: {e}")

            return gpuName
        elif self.jtopLoaded:
            # Access the GPU name from self.jtopInstance.gpu
            try:
                gpu_info = self.jtopInstance.gpu
                gpu_name = next(iter(gpu_info.keys()))
                return gpu_name
            except Exception as e:
                logger.error('Could not get GPU name. ' + str(e))
                return 'Unknown GPU'
        else:
            return ''

    def systemGetDriverVersion(self):
        if self.tpuLoaded:
            try:
                import jax
                return f'JAX version: {jax.__version__}'
            except Exception:
                return 'JAX TPU'
        elif self.pynvmlLoaded:
            return f'NVIDIA Driver: {self.pynvml.nvmlSystemGetDriverVersion()}'
        elif self.jtopLoaded:
            # No direct method to get driver version from jtop
            return 'NVIDIA Driver: unknown'
        else:
            return 'Driver unknown'

    def deviceGetUtilizationRates(self, deviceHandle):
        if self.tpuLoaded:
            # Try to get TensorCore utilization via tpu_info.metrics
            # This only works when TPU is actively being used by a framework
            try:
                from tpu_info import metrics as tpu_metrics
                # Get chip usage (HBM and duty cycle)
                usage = tpu_metrics.get_chip_usage(deviceHandle)
                if usage and hasattr(usage, 'duty_cycle'):
                    # Return duty cycle as percentage
                    return usage.duty_cycle if usage.duty_cycle is not None else -1
            except Exception:
                pass
            # Fallback: TPU metrics unavailable (TPU not actively in use)
            return -1
        elif self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetUtilizationRates(deviceHandle).gpu
        elif self.jtopLoaded:
            # GPU utilization from jtop stats
            try:
                gpu_util = self.jtopInstance.stats.get('GPU', -1)
                return gpu_util
            except Exception as e:
                logger.error('Could not get GPU utilization. ' + str(e))
                return -1
        else:
            return 0

    def deviceGetMemoryInfo(self, deviceHandle):
        if self.tpuLoaded:
            # If deviceHandle is a JAX device, use its memory_stats
            if deviceHandle is not None:
                try:
                    stats = deviceHandle.memory_stats()
                    if stats:
                        total = stats.get('bytes_limit', 0)
                        used = stats.get('bytes_in_use', 0)
                        return {'total': total, 'used': used}
                except Exception as e:
                    logger.debug(f'Could not get JAX TPU memory info: {e}')
            
            # Fallback to tpu_info for memory info
            # (used when JAX init failed but tpu_info is available)
            # Memory info is part of the usage cache
            # We'll get it from _refreshTPUUsageCache in getStatus
            return {'total': self.tpuHbmGib * 1024 * 1024 * 1024, 'used': 0}
        elif self.pynvmlLoaded:
            mem = self.pynvml.nvmlDeviceGetMemoryInfo(deviceHandle)
            return {'total': mem.total, 'used': mem.used}
        elif self.jtopLoaded:
            mem_data = self.jtopInstance.memory['RAM']
            total = mem_data['tot']
            used = mem_data['used']
            return {'total': total, 'used': used}
        else:
            return {'total': 1, 'used': 1}

    def deviceGetTemperature(self, deviceHandle):
        if self.tpuLoaded:
            # TPU doesn't expose temperature via JAX
            return -1
        elif self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetTemperature(deviceHandle, self.pynvml.NVML_TEMPERATURE_GPU)
        elif self.jtopLoaded:
            try:
                temperature = self.jtopInstance.stats.get('Temp gpu', -1)
                return temperature
            except Exception as e:
                logger.error('Could not get GPU temperature. ' + str(e))
                return -1
        else:
            return 0

    def _refreshTPUUsageCache(self):
        """
        Refresh the TPU usage cache from tpu_info.metrics.
        Returns list of Usage objects for all chips.
        """
        import time
        current_time = time.time()
        
        # Cache for 1 second to avoid too many gRPC calls
        if self._tpu_usage_cache is not None and (current_time - self._tpu_usage_cache_time) < 1.0:
            return self._tpu_usage_cache
        
        try:
            from tpu_info import metrics as tpu_metrics
            if self.tpuChipTypeEnum is not None:
                self._tpu_usage_cache = tpu_metrics.get_chip_usage(self.tpuChipTypeEnum)
                self._tpu_usage_cache_time = current_time
                return self._tpu_usage_cache
        except Exception as e:
            logger.debug(f'Could not refresh TPU usage cache: {e}')
        
        return None
    
    def _refreshTensorCoreCache(self):
        """
        Refresh the TensorCore utilization cache from libtpu SDK.
        Returns list of utilization percentages for all cores.
        """
        import time
        current_time = time.time()
        
        # Cache for 1 second
        if self._tpu_tc_cache is not None and (current_time - self._tpu_tc_cache_time) < 1.0:
            return self._tpu_tc_cache
        
        try:
            # Try to import libtpu SDK
            try:
                from libtpu import sdk as libtpu_sdk
            except ImportError:
                import libtpu.sdk as libtpu_sdk
            
            monitoring_module = None
            if hasattr(libtpu_sdk, 'tpumonitoring'):
                monitoring_module = libtpu_sdk.tpumonitoring
            elif hasattr(libtpu_sdk, 'monitoring'):
                monitoring_module = libtpu_sdk.monitoring
            
            if monitoring_module:
                tc_data = monitoring_module.get_metric("tensorcore_util").data()
                if tc_data:
                    self._tpu_tc_cache = tc_data
                    self._tpu_tc_cache_time = current_time
                    return self._tpu_tc_cache
        except ImportError:
            logger.debug('libtpu SDK not available for TensorCore monitoring')
        except Exception as e:
            logger.debug(f'Could not refresh TensorCore cache: {e}')
        
        return None

    def deviceGetDutyCycle(self, deviceHandle, deviceIndex):
        """
        Get TPU Duty Cycle (percentage of time TPU is active).
        This requires libtpu gRPC connection on port 8431.
        Only available when TPU is actively being used by a framework.
        """
        if not self.tpuLoaded:
            logger.debug(f'deviceGetDutyCycle: tpuLoaded is False')
            return -1
        
        try:
            usage_list = self._refreshTPUUsageCache()
            if usage_list and deviceIndex < len(usage_list):
                usage = usage_list[deviceIndex]
                # Use duty_cycle_pct attribute (not duty_cycle)
                duty_cycle_pct = getattr(usage, 'duty_cycle_pct', None)
                if duty_cycle_pct is not None:
                    return float(duty_cycle_pct)
            else:
                logger.debug(f'deviceGetDutyCycle: usage_list is None or index out of range')
        except Exception as e:
            logger.error(f'Could not get TPU Duty Cycle: {e}')
        
        return -1

    def deviceGetTensorCoreUtilization(self, deviceHandle, deviceIndex):
        """
        Get TPU TensorCore Utilization (percentage of TensorCore capacity in use).
        This requires libtpu SDK and vbar control agent.
        Only available when TPU is actively being used by a framework.
        """
        if not self.tpuLoaded:
            return -1
        
        try:
            tc_data = self._refreshTensorCoreCache()
            if tc_data and deviceIndex < len(tc_data):
                return float(tc_data[deviceIndex])
        except Exception as e:
            logger.debug(f'Could not get TPU TensorCore Utilization: {e}')
        
        return -1

    def close(self):
        if self.jtopLoaded and self.jtopInstance is not None:
            self.jtopInstance.close()
