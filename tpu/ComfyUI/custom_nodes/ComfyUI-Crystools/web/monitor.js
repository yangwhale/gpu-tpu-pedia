import { app, api, ComfyButtonGroup } from './comfy/index.js';
import { commonPrefix } from './common.js';
import { MonitorUI } from './monitorUI.js';
import { Colors } from './styles.js';
import { convertNumberToPascalCase } from './utils.js';
import { ComfyKeyMenuDisplayOption, MenuDisplayOptions } from './progressBarUIBase.js';
class CrystoolsMonitor {
    constructor() {
        Object.defineProperty(this, "idExtensionName", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 'Crystools.monitor'
        });
        Object.defineProperty(this, "menuPrefix", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: commonPrefix
        });
        Object.defineProperty(this, "menuDisplayOption", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: MenuDisplayOptions.Disabled
        });
        Object.defineProperty(this, "crystoolsButtonGroup", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: null
        });
        Object.defineProperty(this, "settingsRate", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "settingsMonitorHeight", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "settingsMonitorWidth", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "monitorCPUElement", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "monitorRAMElement", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "monitorHDDElement", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "settingsHDD", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "monitorGPUSettings", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: []
        });
        Object.defineProperty(this, "monitorVRAMSettings", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: []
        });
        Object.defineProperty(this, "monitorTemperatureSettings", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: []
        });
        // TPU specific settings
        Object.defineProperty(this, "monitorDutyCycleSettings", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: []
        });
        Object.defineProperty(this, "monitorTensorCoreSettings", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: []
        });
        Object.defineProperty(this, "monitorUI", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "monitorWidthId", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 'Crystools.MonitorWidth'
        });
        Object.defineProperty(this, "monitorWidth", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 100
        });
        Object.defineProperty(this, "monitorHeightId", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 'Crystools.MonitorHeight'
        });
        Object.defineProperty(this, "monitorHeight", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 18
        });
        Object.defineProperty(this, "createSettingsRate", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                this.settingsRate = {
                    id: 'Crystools.RefreshRate',
                    name: 'Refresh per second',
                    category: ['Crystools', this.menuPrefix + ' Configuration', 'refresh'],
                    tooltip: 'This is the time (in seconds) between each update of the monitors, 0 means no refresh',
                    type: 'slider',
                    attrs: {
                        min: 0,
                        max: 2,
                        step: .25,
                    },
                    defaultValue: .5,
                    onChange: async (value) => {
                        let valueNumber;
                        try {
                            valueNumber = parseFloat(value);
                            if (isNaN(valueNumber)) {
                                throw new Error('invalid value');
                            }
                        }
                        catch (error) {
                            console.error(error);
                            return;
                        }
                        try {
                            await this.updateServer({ rate: valueNumber });
                        }
                        catch (error) {
                            console.error(error);
                            return;
                        }
                        const data = {
                            cpu_utilization: 0,
                            device: 'cpu',
                            gpus: [
                                {
                                    gpu_utilization: 0,
                                    gpu_temperature: 0,
                                    vram_total: 0,
                                    vram_used: 0,
                                    vram_used_percent: 0,
                                },
                            ],
                            hdd_total: 0,
                            hdd_used: 0,
                            hdd_used_percent: 0,
                            ram_total: 0,
                            ram_used: 0,
                            ram_used_percent: 0,
                        };
                        if (valueNumber === 0) {
                            this.monitorUI.updateDisplay(data);
                        }
                        this.monitorUI?.updateAllAnimationDuration(valueNumber);
                    },
                };
            }
        });
        Object.defineProperty(this, "createSettingsMonitorWidth", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                this.settingsMonitorWidth = {
                    id: this.monitorWidthId,
                    name: 'Pixel Width',
                    category: ['Crystools', this.menuPrefix + ' Configuration', 'width'],
                    tooltip: 'The width of the monitor in pixels on the UI (only on top/bottom UI)',
                    type: 'slider',
                    attrs: {
                        min: 60,
                        max: 100,
                        step: 1,
                    },
                    defaultValue: this.monitorWidth,
                    onChange: (value) => {
                        let valueNumber;
                        try {
                            valueNumber = parseInt(value);
                            if (isNaN(valueNumber)) {
                                throw new Error('invalid value');
                            }
                        }
                        catch (error) {
                            console.error(error);
                            return;
                        }
                        const h = app.extensionManager.setting.get(this.monitorHeightId);
                        this.monitorUI?.updateMonitorSize(valueNumber, h);
                    },
                };
            }
        });
        Object.defineProperty(this, "createSettingsMonitorHeight", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                this.settingsMonitorHeight = {
                    id: this.monitorHeightId,
                    name: 'Pixel Height',
                    category: ['Crystools', this.menuPrefix + ' Configuration', 'height'],
                    tooltip: 'The height of the monitor in pixels on the UI (only on top/bottom UI)',
                    type: 'slider',
                    attrs: {
                        min: 16,
                        max: 50,
                        step: 1,
                    },
                    defaultValue: this.monitorHeight,
                    onChange: async (value) => {
                        let valueNumber;
                        try {
                            valueNumber = parseInt(value);
                            if (isNaN(valueNumber)) {
                                throw new Error('invalid value');
                            }
                        }
                        catch (error) {
                            console.error(error);
                            return;
                        }
                        const w = await app.extensionManager.setting.get(this.monitorWidthId);
                        this.monitorUI?.updateMonitorSize(w, valueNumber);
                    },
                };
            }
        });
        Object.defineProperty(this, "createSettingsCPU", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                this.monitorCPUElement = {
                    id: 'Crystools.ShowCpu',
                    name: 'CPU Usage',
                    category: ['Crystools', this.menuPrefix + ' Hardware', 'Cpu'],
                    type: 'boolean',
                    label: 'CPU',
                    symbol: '%',
                    defaultValue: true,
                    htmlMonitorRef: undefined,
                    htmlMonitorSliderRef: undefined,
                    htmlMonitorLabelRef: undefined,
                    cssColor: Colors.CPU,
                    onChange: async (value) => {
                        await this.updateServer({ switchCPU: value });
                        this.updateWidget(this.monitorCPUElement);
                    },
                };
            }
        });
        Object.defineProperty(this, "createSettingsRAM", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                this.monitorRAMElement = {
                    id: 'Crystools.ShowRam',
                    name: 'RAM Used',
                    category: ['Crystools', this.menuPrefix + ' Hardware', 'Ram'],
                    type: 'boolean',
                    label: 'RAM',
                    symbol: '%',
                    defaultValue: true,
                    htmlMonitorRef: undefined,
                    htmlMonitorSliderRef: undefined,
                    htmlMonitorLabelRef: undefined,
                    cssColor: Colors.RAM,
                    onChange: async (value) => {
                        await this.updateServer({ switchRAM: value });
                        this.updateWidget(this.monitorRAMElement);
                    },
                };
            }
        });
        Object.defineProperty(this, "deviceType", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 'cuda'
        });
        Object.defineProperty(this, "createSettingsGPUUsage", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (name, index, moreThanOneGPU) => {
                if (name === undefined || index === undefined) {
                    console.warn('getGPUsFromServer: name or index undefined', name, index);
                    return;
                }
                // For TPU, skip GPU usage (not supported)
                if (this.deviceType === 'tpu') {
                    return;
                }
                let label = 'GPU ';
                label += moreThanOneGPU ? index : '';
                const monitorGPUNElement = {
                    id: 'Crystools.ShowGpuUsage' + convertNumberToPascalCase(index),
                    name: ' Usage',
                    category: ['Crystools', `${this.menuPrefix} Show GPU [${index}] ${name}`, 'Usage'],
                    type: 'boolean',
                    label,
                    symbol: '%',
                    monitorTitle: `${index}: ${name}`,
                    defaultValue: true,
                    htmlMonitorRef: undefined,
                    htmlMonitorSliderRef: undefined,
                    htmlMonitorLabelRef: undefined,
                    cssColor: Colors.GPU,
                    onChange: async (value) => {
                        await this.updateServerGPU(index, { utilization: value });
                        this.updateWidget(monitorGPUNElement);
                    },
                };
                this.monitorGPUSettings[index] = monitorGPUNElement;
                app.ui.settings.addSetting(this.monitorGPUSettings[index]);
                this.monitorUI.createDOMGPUMonitor(this.monitorGPUSettings[index]);
            }
        });
        Object.defineProperty(this, "createSettingsGPUVRAM", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (name, index, moreThanOneGPU) => {
                if (name === undefined || index === undefined) {
                    console.warn('getGPUsFromServer: name or index undefined', name, index);
                    return;
                }
                // Use TPU/HBM labels for TPU
                const isTPU = this.deviceType === 'tpu';
                let label = isTPU ? 'HBM ' : 'VRAM ';
                label += moreThanOneGPU ? index : '';
                const categoryName = isTPU ? `${this.menuPrefix} Show TPU [${index}]` : `${this.menuPrefix} Show GPU [${index}] ${name}`;
                const memLabel = isTPU ? 'HBM' : 'VRAM';
                const monitorVRAMNElement = {
                    id: 'Crystools.ShowGpuVram' + convertNumberToPascalCase(index),
                    name: memLabel,
                    category: ['Crystools', categoryName, memLabel],
                    type: 'boolean',
                    label: label,
                    symbol: '%',
                    monitorTitle: isTPU ? `TPU ${index}` : `${index}: ${name}`,
                    defaultValue: true,
                    htmlMonitorRef: undefined,
                    htmlMonitorSliderRef: undefined,
                    htmlMonitorLabelRef: undefined,
                    cssColor: Colors.VRAM,
                    onChange: async (value) => {
                        await this.updateServerGPU(index, { vram: value });
                        this.updateWidget(monitorVRAMNElement);
                    },
                };
                this.monitorVRAMSettings[index] = monitorVRAMNElement;
                app.ui.settings.addSetting(this.monitorVRAMSettings[index]);
                // For TPU, use the grouped layout
                if (isTPU) {
                    this.monitorUI.createDOMTPUMonitor(this.monitorVRAMSettings[index], index);
                } else {
                    this.monitorUI.createDOMGPUMonitor(this.monitorVRAMSettings[index]);
                }
            }
        });
        Object.defineProperty(this, "createSettingsGPUTemp", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (name, index, moreThanOneGPU) => {
                if (name === undefined || index === undefined) {
                    console.warn('getGPUsFromServer: name or index undefined', name, index);
                    return;
                }
                // For TPU, skip temperature (not supported)
                if (this.deviceType === 'tpu') {
                    return;
                }
                let label = 'Temp ';
                label += moreThanOneGPU ? index : '';
                const monitorTemperatureNElement = {
                    id: 'Crystools.ShowGpuTemperature' + convertNumberToPascalCase(index),
                    name: 'Temperature',
                    category: ['Crystools', `${this.menuPrefix} Show GPU [${index}] ${name}`, 'Temperature'],
                    type: 'boolean',
                    label: label,
                    symbol: '°',
                    monitorTitle: `${index}: ${name}`,
                    defaultValue: true,
                    htmlMonitorRef: undefined,
                    htmlMonitorSliderRef: undefined,
                    htmlMonitorLabelRef: undefined,
                    cssColor: Colors.TEMP_START,
                    cssColorFinal: Colors.TEMP_END,
                    onChange: async (value) => {
                        await this.updateServerGPU(index, { temperature: value });
                        this.updateWidget(monitorTemperatureNElement);
                    },
                };
                this.monitorTemperatureSettings[index] = monitorTemperatureNElement;
                app.ui.settings.addSetting(this.monitorTemperatureSettings[index]);
                this.monitorUI.createDOMGPUMonitor(this.monitorTemperatureSettings[index]);
            }
        });
        // TPU specific: Duty Cycle setting
        Object.defineProperty(this, "createSettingsTPUDutyCycle", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (name, index, moreThanOneTPU) => {
                if (name === undefined || index === undefined) {
                    console.warn('createSettingsTPUDutyCycle: name or index undefined', name, index);
                    return;
                }
                // Only for TPU
                if (this.deviceType !== 'tpu') {
                    return;
                }
                let label = 'Busy ';
                label += moreThanOneTPU ? index : '';
                const monitorDutyCycleElement = {
                    id: 'Crystools.ShowTpuDutyCycle' + convertNumberToPascalCase(index),
                    name: 'Duty Cycle',
                    category: ['Crystools', `${this.menuPrefix} Show TPU [${index}]`, 'Duty Cycle'],
                    type: 'boolean',
                    label: label,
                    symbol: '%',
                    monitorTitle: `TPU ${index} Duty Cycle`,
                    defaultValue: true,
                    htmlMonitorRef: undefined,
                    htmlMonitorSliderRef: undefined,
                    htmlMonitorLabelRef: undefined,
                    cssColor: Colors.DUTY_CYCLE,
                    onChange: async (value) => {
                        await this.updateServerGPU(index, { duty_cycle: value });
                        this.updateWidget(monitorDutyCycleElement);
                    },
                };
                this.monitorDutyCycleSettings[index] = monitorDutyCycleElement;
                app.ui.settings.addSetting(this.monitorDutyCycleSettings[index]);
                this.monitorUI.createDOMTPUMonitor(this.monitorDutyCycleSettings[index], index);
            }
        });
        // TPU specific: TensorCore Utilization setting
        Object.defineProperty(this, "createSettingsTPUTensorCore", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (name, index, moreThanOneTPU) => {
                if (name === undefined || index === undefined) {
                    console.warn('createSettingsTPUTensorCore: name or index undefined', name, index);
                    return;
                }
                // Only for TPU
                if (this.deviceType !== 'tpu') {
                    return;
                }
                let label = 'MFU ';
                label += moreThanOneTPU ? index : '';
                const monitorTensorCoreElement = {
                    id: 'Crystools.ShowTpuTensorCore' + convertNumberToPascalCase(index),
                    name: 'TensorCore Util',
                    category: ['Crystools', `${this.menuPrefix} Show TPU [${index}]`, 'TensorCore'],
                    type: 'boolean',
                    label: label,
                    symbol: '%',
                    monitorTitle: `TPU ${index} TensorCore`,
                    defaultValue: true,
                    htmlMonitorRef: undefined,
                    htmlMonitorSliderRef: undefined,
                    htmlMonitorLabelRef: undefined,
                    cssColor: Colors.TENSORCORE,
                    onChange: async (value) => {
                        await this.updateServerGPU(index, { tensorcore: value });
                        this.updateWidget(monitorTensorCoreElement);
                    },
                };
                this.monitorTensorCoreSettings[index] = monitorTensorCoreElement;
                app.ui.settings.addSetting(this.monitorTensorCoreSettings[index]);
                this.monitorUI.createDOMTPUMonitor(this.monitorTensorCoreSettings[index], index);
            }
        });
        Object.defineProperty(this, "createSettingsHDD", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                this.monitorHDDElement = {
                    id: 'Crystools.ShowHdd',
                    name: 'Show HDD Used',
                    category: ['Crystools', this.menuPrefix + ' Show Hard Disk', 'Show'],
                    type: 'boolean',
                    label: 'HDD',
                    symbol: '%',
                    defaultValue: false,
                    htmlMonitorRef: undefined,
                    htmlMonitorSliderRef: undefined,
                    htmlMonitorLabelRef: undefined,
                    cssColor: Colors.DISK,
                    onChange: async (value) => {
                        await this.updateServer({ switchHDD: value });
                        this.updateWidget(this.monitorHDDElement);
                    },
                };
                this.settingsHDD = {
                    id: 'Crystools.WhichHdd',
                    name: 'Partition to show',
                    category: ['Crystools', this.menuPrefix + ' Show Hard Disk', 'Which'],
                    type: 'combo',
                    defaultValue: '/',
                    options: [],
                    onChange: async (value) => {
                        await this.updateServer({ whichHDD: value });
                    },
                };
            }
        });
        Object.defineProperty(this, "createSettings", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                app.ui.settings.addSetting(this.settingsRate);
                app.ui.settings.addSetting(this.settingsMonitorHeight);
                app.ui.settings.addSetting(this.settingsMonitorWidth);
                app.ui.settings.addSetting(this.monitorRAMElement);
                app.ui.settings.addSetting(this.monitorCPUElement);
                void this.getHDDsFromServer().then((data) => {
                    this.settingsHDD.options = data;
                    app.ui.settings.addSetting(this.settingsHDD);
                });
                app.ui.settings.addSetting(this.monitorHDDElement);
                // First get device type, then GPUs
                void this.getDeviceTypeFromServer().then((deviceType) => {
                    this.deviceType = deviceType || 'cuda';
                    console.log('[Crystools] Device type:', this.deviceType);
                    return this.getGPUsFromServer();
                }).then((gpus) => {
                    let moreThanOne = false;
                    if (gpus.length > 1) {
                        moreThanOne = true;
                    }
                    // For TPU, only show first 2 devices to save space
                    const maxTPUDisplay = 2;
                    const displayGpus = this.deviceType === 'tpu' ? gpus.slice(0, maxTPUDisplay) : gpus;
                    
                    displayGpus?.forEach(({ name, index }, arrayIndex) => {
                        this.createSettingsGPUTemp(name, index, moreThanOne);
                        this.createSettingsGPUVRAM(name, index, moreThanOne);
                        this.createSettingsGPUUsage(name, index, moreThanOne);
                        // TPU specific: Duty Cycle and TensorCore
                        if (this.deviceType === 'tpu') {
                            this.createSettingsTPUDutyCycle(name, index, moreThanOne);
                            this.createSettingsTPUTensorCore(name, index, moreThanOne);
                            // TPU monitors are now grouped in vertical containers
                        }
                    });
                    this.finishedLoad();
                });
            }
        });
        Object.defineProperty(this, "finishedLoad", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                this.monitorUI.orderMonitors();
                this.updateAllWidget();
                this.moveMonitor(this.menuDisplayOption);
                const w = app.extensionManager.setting.get(this.monitorWidthId);
                const h = app.extensionManager.setting.get(this.monitorHeightId);
                this.monitorUI.updateMonitorSize(w, h);
            }
        });
        Object.defineProperty(this, "updateDisplay", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (value) => {
                if (value !== this.menuDisplayOption) {
                    this.menuDisplayOption = value;
                    this.moveMonitor(this.menuDisplayOption);
                }
            }
        });
        Object.defineProperty(this, "moveMonitor", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (menuPosition) => {
                let parentElement;
                switch (menuPosition) {
                    case MenuDisplayOptions.Disabled:
                        parentElement = document.getElementById('queue-button');
                        if (parentElement && this.monitorUI.rootElement) {
                            parentElement.insertAdjacentElement('afterend', this.crystoolsButtonGroup.element);
                        }
                        else {
                            console.error('Crystools: parentElement to move monitors not found!', parentElement);
                        }
                        break;
                    case MenuDisplayOptions.Top:
                    case MenuDisplayOptions.Bottom:
                        app.menu?.settingsGroup.element.before(this.crystoolsButtonGroup.element);
                }
            }
        });
        Object.defineProperty(this, "updateAllWidget", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                this.updateWidget(this.monitorCPUElement);
                this.updateWidget(this.monitorRAMElement);
                this.updateWidget(this.monitorHDDElement);
                this.monitorGPUSettings.forEach((monitorSettings) => {
                    monitorSettings && this.updateWidget(monitorSettings);
                });
                this.monitorVRAMSettings.forEach((monitorSettings) => {
                    monitorSettings && this.updateWidget(monitorSettings);
                });
                this.monitorTemperatureSettings.forEach((monitorSettings) => {
                    monitorSettings && this.updateWidget(monitorSettings);
                });
                // TPU specific widgets
                this.monitorDutyCycleSettings.forEach((monitorSettings) => {
                    monitorSettings && this.updateWidget(monitorSettings);
                });
                this.monitorTensorCoreSettings.forEach((monitorSettings) => {
                    monitorSettings && this.updateWidget(monitorSettings);
                });
            }
        });
        Object.defineProperty(this, "updateWidget", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (monitorSettings) => {
                if (this.monitorUI) {
                    const value = app.extensionManager.setting.get(monitorSettings.id);
                    this.monitorUI.showMonitor(monitorSettings, value);
                }
            }
        });
        Object.defineProperty(this, "updateServer", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: async (data) => {
                const resp = await api.fetchApi('/crystools/monitor', {
                    method: 'PATCH',
                    body: JSON.stringify(data),
                    cache: 'no-store',
                });
                if (resp.status === 200) {
                    return await resp.text();
                }
                throw new Error(resp.statusText);
            }
        });
        Object.defineProperty(this, "updateServerGPU", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: async (index, data) => {
                const resp = await api.fetchApi(`/crystools/monitor/GPU/${index}`, {
                    method: 'PATCH',
                    body: JSON.stringify(data),
                    cache: 'no-store',
                });
                if (resp.status === 200) {
                    return await resp.text();
                }
                throw new Error(resp.statusText);
            }
        });
        Object.defineProperty(this, "getHDDsFromServer", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: async () => {
                return this.getDataFromServer('HDD');
            }
        });
        Object.defineProperty(this, "getGPUsFromServer", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: async () => {
                return this.getDataFromServer('GPU');
            }
        });
        Object.defineProperty(this, "getDeviceTypeFromServer", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: async () => {
                try {
                    const resp = await api.fetchApi('/crystools/monitor/device_type', {
                        method: 'GET',
                        cache: 'no-store',
                    });
                    if (resp.status === 200) {
                        const data = await resp.json();
                        return data.device_type || 'cuda';
                    }
                } catch (e) {
                    console.warn('[Crystools] Could not get device type:', e);
                }
                return 'cuda';
            }
        });
        Object.defineProperty(this, "getDataFromServer", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: async (what) => {
                const resp = await api.fetchApi(`/crystools/monitor/${what}`, {
                    method: 'GET',
                    cache: 'no-store',
                });
                if (resp.status === 200) {
                    return await resp.json();
                }
                throw new Error(resp.statusText);
            }
        });
        Object.defineProperty(this, "setup", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                if (this.monitorUI) {
                    return;
                }
                this.createSettingsRate();
                this.createSettingsMonitorHeight();
                this.createSettingsMonitorWidth();
                this.createSettingsCPU();
                this.createSettingsRAM();
                this.createSettingsHDD();
                this.createSettings();
                const currentRate = parseFloat(app.extensionManager.setting.get(this.settingsRate.id));
                this.menuDisplayOption = app.extensionManager.setting.get(ComfyKeyMenuDisplayOption);
                app.ui.settings.addEventListener(`${ComfyKeyMenuDisplayOption}.change`, (e) => {
                    this.updateDisplay(e.detail.value);
                });
                this.crystoolsButtonGroup = new ComfyButtonGroup();
                app.menu?.settingsGroup.element.before(this.crystoolsButtonGroup.element);
                this.monitorUI = new MonitorUI(this.crystoolsButtonGroup.element, this.monitorCPUElement, this.monitorRAMElement, this.monitorHDDElement, this.monitorGPUSettings, this.monitorVRAMSettings, this.monitorTemperatureSettings, currentRate, this.monitorDutyCycleSettings, this.monitorTensorCoreSettings);
                this.updateDisplay(this.menuDisplayOption);
                this.registerListeners();
            }
        });
        Object.defineProperty(this, "registerListeners", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                api.addEventListener('crystools.monitor', (event) => {
                    if (event?.detail === undefined) {
                        return;
                    }
                    this.monitorUI.updateDisplay(event.detail);
                }, false);
            }
        });
    }
}
const crystoolsMonitor = new CrystoolsMonitor();
app.registerExtension({
    name: crystoolsMonitor.idExtensionName,
    setup: crystoolsMonitor.setup,
});
