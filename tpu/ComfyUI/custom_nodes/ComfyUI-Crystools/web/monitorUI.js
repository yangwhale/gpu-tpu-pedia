import { ProgressBarUIBase } from './progressBarUIBase.js';
import { createStyleSheet, formatBytes } from './utils.js';

// Format bytes to GiB with 2 decimal places (e.g., "18.95")
function formatGiB(bytes) {
    if (bytes === 0 || bytes === undefined) return '0';
    const gib = bytes / (1024 * 1024 * 1024);
    return gib.toFixed(2);
}

export class MonitorUI extends ProgressBarUIBase {
    constructor(rootElement, monitorCPUElement, monitorRAMElement, monitorHDDElement, monitorGPUSettings, monitorVRAMSettings, monitorTemperatureSettings, currentRate, monitorDutyCycleSettings = [], monitorTensorCoreSettings = []) {
        super('crystools-monitors-root', rootElement);
        Object.defineProperty(this, "rootElement", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: rootElement
        });
        Object.defineProperty(this, "monitorCPUElement", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: monitorCPUElement
        });
        Object.defineProperty(this, "monitorRAMElement", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: monitorRAMElement
        });
        Object.defineProperty(this, "monitorHDDElement", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: monitorHDDElement
        });
        Object.defineProperty(this, "monitorGPUSettings", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: monitorGPUSettings
        });
        Object.defineProperty(this, "monitorVRAMSettings", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: monitorVRAMSettings
        });
        Object.defineProperty(this, "monitorTemperatureSettings", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: monitorTemperatureSettings
        });
        // TPU specific settings
        Object.defineProperty(this, "monitorDutyCycleSettings", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: monitorDutyCycleSettings
        });
        Object.defineProperty(this, "monitorTensorCoreSettings", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: monitorTensorCoreSettings
        });
        Object.defineProperty(this, "currentRate", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: currentRate
        });
        Object.defineProperty(this, "lastMonitor", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 1
        });
        Object.defineProperty(this, "styleSheet", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "maxVRAMUsed", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {}
        });
        Object.defineProperty(this, "rowBreakElements", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: []
        });
        Object.defineProperty(this, "tpuGroupContainers", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: {}
        });
        Object.defineProperty(this, "tpuWrapper", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: null
        });
        Object.defineProperty(this, "cpuRamContainer", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: null
        });
        Object.defineProperty(this, "createDOM", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                if (!this.rootElement) {
                    throw Error('Crystools: MonitorUI - Container not found');
                }
                // Create a vertical container for CPU and RAM
                this.cpuRamContainer = document.createElement('div');
                this.cpuRamContainer.classList.add('crystools-cpu-ram-container');
                this.cpuRamContainer.appendChild(this.createMonitor(this.monitorCPUElement));
                this.cpuRamContainer.appendChild(this.createMonitor(this.monitorRAMElement));
                this.rootElement.appendChild(this.cpuRamContainer);
                this.rootElement.appendChild(this.createMonitor(this.monitorHDDElement));
                this.updateAllAnimationDuration(this.currentRate);
            }
        });
        Object.defineProperty(this, "createDOMGPUMonitor", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (monitorSettings) => {
                if (!(monitorSettings && this.rootElement)) {
                    return;
                }
                this.rootElement.appendChild(this.createMonitor(monitorSettings));
                this.updateAllAnimationDuration(this.currentRate);
            }
        });
        Object.defineProperty(this, "createRowBreak", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (afterIndex) => {
                if (!this.rootElement) {
                    return;
                }
                const rowBreak = document.createElement('div');
                rowBreak.classList.add('crystools-tpu-row-break');
                this.rootElement.appendChild(rowBreak);
                this.rowBreakElements[afterIndex] = rowBreak;
            }
        });
        Object.defineProperty(this, "getOrCreateTPUWrapper", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                if (!this.tpuWrapper) {
                    this.tpuWrapper = document.createElement('div');
                    this.tpuWrapper.classList.add('crystools-tpu-wrapper');
                    if (this.rootElement) {
                        this.rootElement.appendChild(this.tpuWrapper);
                    }
                }
                return this.tpuWrapper;
            }
        });
        Object.defineProperty(this, "getOrCreateTPUGroup", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (tpuIndex) => {
                if (!this.tpuGroupContainers[tpuIndex]) {
                    const container = document.createElement('div');
                    container.classList.add('crystools-tpu-group');
                    container.dataset.tpuIndex = tpuIndex;
                    this.tpuGroupContainers[tpuIndex] = container;
                    // Add to TPU wrapper instead of root
                    const wrapper = this.getOrCreateTPUWrapper();
                    wrapper.appendChild(container);
                }
                return this.tpuGroupContainers[tpuIndex];
            }
        });
        Object.defineProperty(this, "createDOMTPUMonitor", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (monitorSettings, tpuIndex) => {
                if (!(monitorSettings && this.rootElement)) {
                    return;
                }
                const tpuGroup = this.getOrCreateTPUGroup(tpuIndex);
                tpuGroup.appendChild(this.createMonitor(monitorSettings));
                this.updateAllAnimationDuration(this.currentRate);
            }
        });
        Object.defineProperty(this, "orderMonitors", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                try {
                    this.monitorCPUElement.htmlMonitorRef.style.order = '' + this.lastMonitor++;
                    this.monitorRAMElement.htmlMonitorRef.style.order = '' + this.lastMonitor++;
                    this.monitorGPUSettings.forEach((_monitorSettings, index) => {
                        if (this.monitorGPUSettings[index]?.htmlMonitorRef) {
                            this.monitorGPUSettings[index].htmlMonitorRef.style.order = '' + this.lastMonitor++;
                        }
                        if (this.monitorVRAMSettings[index]?.htmlMonitorRef) {
                            this.monitorVRAMSettings[index].htmlMonitorRef.style.order = '' + this.lastMonitor++;
                        }
                        if (this.monitorTemperatureSettings[index]?.htmlMonitorRef) {
                            this.monitorTemperatureSettings[index].htmlMonitorRef.style.order = '' + this.lastMonitor++;
                        }
                        // TPU specific monitors
                        if (this.monitorDutyCycleSettings[index]?.htmlMonitorRef) {
                            this.monitorDutyCycleSettings[index].htmlMonitorRef.style.order = '' + this.lastMonitor++;
                        }
                        if (this.monitorTensorCoreSettings[index]?.htmlMonitorRef) {
                            this.monitorTensorCoreSettings[index].htmlMonitorRef.style.order = '' + this.lastMonitor++;
                        }
                        // Row break after this TPU group
                        if (this.rowBreakElements[index]) {
                            this.rowBreakElements[index].style.order = '' + this.lastMonitor++;
                        }
                    });
                    // Also check for TPU monitors if GPU settings are empty
                    if (this.monitorGPUSettings.length === 0 && this.monitorVRAMSettings.length > 0) {
                        this.monitorVRAMSettings.forEach((_monitorSettings, index) => {
                            if (this.monitorVRAMSettings[index]?.htmlMonitorRef) {
                                this.monitorVRAMSettings[index].htmlMonitorRef.style.order = '' + this.lastMonitor++;
                            }
                            if (this.monitorDutyCycleSettings[index]?.htmlMonitorRef) {
                                this.monitorDutyCycleSettings[index].htmlMonitorRef.style.order = '' + this.lastMonitor++;
                            }
                            if (this.monitorTensorCoreSettings[index]?.htmlMonitorRef) {
                                this.monitorTensorCoreSettings[index].htmlMonitorRef.style.order = '' + this.lastMonitor++;
                            }
                            // Row break after this TPU group
                            if (this.rowBreakElements[index]) {
                                this.rowBreakElements[index].style.order = '' + this.lastMonitor++;
                            }
                        });
                    }
                    this.monitorHDDElement.htmlMonitorRef.style.order = '' + this.lastMonitor++;
                }
                catch (error) {
                    console.error('orderMonitors', error);
                }
            }
        });
        Object.defineProperty(this, "updateDisplay", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (data) => {
                this.updateMonitor(this.monitorCPUElement, data.cpu_utilization);
                this.updateMonitor(this.monitorRAMElement, data.ram_used_percent, data.ram_used, data.ram_total);
                this.updateMonitor(this.monitorHDDElement, data.hdd_used_percent, data.hdd_used, data.hdd_total);
                if (data.gpus === undefined || data.gpus.length === 0) {
                    console.warn('UpdateAllMonitors: no GPU data');
                    return;
                }
                this.monitorGPUSettings.forEach((monitorSettings, index) => {
                    if (data.gpus[index]) {
                        const gpu = data.gpus[index];
                        if (gpu === undefined) {
                            return;
                        }
                        this.updateMonitor(monitorSettings, gpu.gpu_utilization);
                    }
                    else {
                    }
                });
                this.monitorVRAMSettings.forEach((monitorSettings, index) => {
                    if (data.gpus[index]) {
                        const gpu = data.gpus[index];
                        if (gpu === undefined) {
                            return;
                        }
                        this.updateMonitor(monitorSettings, gpu.vram_used_percent, gpu.vram_used, gpu.vram_total);
                    }
                    else {
                    }
                });
                this.monitorTemperatureSettings.forEach((monitorSettings, index) => {
                    if (data.gpus[index]) {
                        const gpu = data.gpus[index];
                        if (gpu === undefined) {
                            return;
                        }
                        this.updateMonitor(monitorSettings, gpu.gpu_temperature);
                        if (monitorSettings.cssColorFinal && monitorSettings.htmlMonitorSliderRef) {
                            monitorSettings.htmlMonitorSliderRef.style.backgroundColor =
                                `color-mix(in srgb, ${monitorSettings.cssColorFinal} ${gpu.gpu_temperature}%, ${monitorSettings.cssColor})`;
                        }
                    }
                    else {
                    }
                });
                // TPU specific: Duty Cycle
                this.monitorDutyCycleSettings.forEach((monitorSettings, index) => {
                    if (monitorSettings && data.gpus[index]) {
                        const gpu = data.gpus[index];
                        if (gpu === undefined || gpu.duty_cycle === undefined) {
                            return;
                        }
                        this.updateMonitor(monitorSettings, gpu.duty_cycle);
                    }
                });
                // TPU specific: TensorCore Utilization
                this.monitorTensorCoreSettings.forEach((monitorSettings, index) => {
                    if (monitorSettings && data.gpus[index]) {
                        const gpu = data.gpus[index];
                        if (gpu === undefined || gpu.tensorcore_util === undefined) {
                            return;
                        }
                        this.updateMonitor(monitorSettings, gpu.tensorcore_util);
                    }
                });
            }
        });
        Object.defineProperty(this, "updateMonitor", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (monitorSettings, percent, used, total) => {
                if (!(monitorSettings.htmlMonitorSliderRef && monitorSettings.htmlMonitorLabelRef)) {
                    return;
                }
                if (percent < 0) {
                    return;
                }
                const prefix = monitorSettings.monitorTitle ? monitorSettings.monitorTitle + ' - ' : '';
                let title = `${Math.floor(percent)}${monitorSettings.symbol}`;
                let postfix = '';
                if (used !== undefined && total !== undefined) {
                    const gpuIndex = parseInt(monitorSettings.monitorTitle?.split(':')[0] || '0');
                    if (!this.maxVRAMUsed[gpuIndex] || this.maxVRAMUsed[gpuIndex] > total) {
                        this.maxVRAMUsed[gpuIndex] = 0;
                    }
                    if (used > this.maxVRAMUsed[gpuIndex]) {
                        this.maxVRAMUsed[gpuIndex] = used;
                    }
                    postfix = ` - ${formatBytes(used)} / ${formatBytes(total)}`;
                    postfix += ` Max: ${formatBytes(this.maxVRAMUsed[gpuIndex])}`;
                }
                title = `${prefix}${title}${postfix}`;
                if (monitorSettings.htmlMonitorRef) {
                    monitorSettings.htmlMonitorRef.title = title;
                }
                // Only for HBM (TPU memory) monitors, show actual usage in GiB format
                // Check if label contains 'HBM' to distinguish from RAM
                const isHBM = monitorSettings.label && monitorSettings.label.includes('HBM');
                if (isHBM && used !== undefined && total !== undefined && total > 0) {
                    monitorSettings.htmlMonitorLabelRef.innerHTML = `${formatGiB(used)}/${formatGiB(total)}`;
                } else {
                    monitorSettings.htmlMonitorLabelRef.innerHTML = `${Math.floor(percent)}${monitorSettings.symbol}`;
                }
                monitorSettings.htmlMonitorSliderRef.style.width = `${Math.floor(percent)}%`;
            }
        });
        Object.defineProperty(this, "updateAllAnimationDuration", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (value) => {
                this.updatedAnimationDuration(this.monitorCPUElement, value);
                this.updatedAnimationDuration(this.monitorRAMElement, value);
                this.updatedAnimationDuration(this.monitorHDDElement, value);
                this.monitorGPUSettings.forEach((monitorSettings) => {
                    monitorSettings && this.updatedAnimationDuration(monitorSettings, value);
                });
                this.monitorVRAMSettings.forEach((monitorSettings) => {
                    monitorSettings && this.updatedAnimationDuration(monitorSettings, value);
                });
                this.monitorTemperatureSettings.forEach((monitorSettings) => {
                    monitorSettings && this.updatedAnimationDuration(monitorSettings, value);
                });
                // TPU specific monitors
                this.monitorDutyCycleSettings.forEach((monitorSettings) => {
                    monitorSettings && this.updatedAnimationDuration(monitorSettings, value);
                });
                this.monitorTensorCoreSettings.forEach((monitorSettings) => {
                    monitorSettings && this.updatedAnimationDuration(monitorSettings, value);
                });
            }
        });
        Object.defineProperty(this, "updatedAnimationDuration", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (monitorSettings, value) => {
                const slider = monitorSettings.htmlMonitorSliderRef;
                if (!slider) {
                    return;
                }
                slider.style.transition = `width ${value.toFixed(1)}s`;
            }
        });
        Object.defineProperty(this, "createMonitor", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (monitorSettings) => {
                if (!monitorSettings) {
                    return document.createElement('div');
                }
                const htmlMain = document.createElement('div');
                htmlMain.classList.add(monitorSettings.id);
                htmlMain.classList.add('crystools-monitor');
                monitorSettings.htmlMonitorRef = htmlMain;
                if (monitorSettings.title) {
                    htmlMain.title = monitorSettings.title;
                }
                const htmlMonitorText = document.createElement('div');
                htmlMonitorText.classList.add('crystools-text');
                htmlMonitorText.innerHTML = monitorSettings.label;
                htmlMain.append(htmlMonitorText);
                const htmlMonitorContent = document.createElement('div');
                htmlMonitorContent.classList.add('crystools-content');
                htmlMain.append(htmlMonitorContent);
                const htmlMonitorSlider = document.createElement('div');
                htmlMonitorSlider.classList.add('crystools-slider');
                if (monitorSettings.cssColorFinal) {
                    htmlMonitorSlider.style.backgroundColor =
                        `color-mix(in srgb, ${monitorSettings.cssColorFinal} 0%, ${monitorSettings.cssColor})`;
                }
                else {
                    htmlMonitorSlider.style.backgroundColor = monitorSettings.cssColor;
                }
                monitorSettings.htmlMonitorSliderRef = htmlMonitorSlider;
                htmlMonitorContent.append(htmlMonitorSlider);
                const htmlMonitorLabel = document.createElement('div');
                htmlMonitorLabel.classList.add('crystools-label');
                monitorSettings.htmlMonitorLabelRef = htmlMonitorLabel;
                htmlMonitorContent.append(htmlMonitorLabel);
                htmlMonitorLabel.innerHTML = '0%';
                return monitorSettings.htmlMonitorRef;
            }
        });
        Object.defineProperty(this, "updateMonitorSize", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (width, height) => {
                this.styleSheet.innerText = `#crystools-monitors-root .crystools-monitor .crystools-content {height: ${height}px; width: ${width}px;}`;
            }
        });
        Object.defineProperty(this, "showMonitor", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (monitorSettings, value) => {
                if (monitorSettings.htmlMonitorRef) {
                    monitorSettings.htmlMonitorRef.style.display = value ? 'flex' : 'none';
                }
            }
        });
        Object.defineProperty(this, "resetMaxVRAM", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                this.maxVRAMUsed = {};
            }
        });
        this.createDOM();
        this.styleSheet = createStyleSheet('crystools-monitors-size');
    }
}
