import { utils } from './comfy/index.js';
utils.addStylesheet('extensions/ComfyUI-Crystools/monitor.css');
export var Styles;
(function (Styles) {
    Styles["BARS"] = "BARS";
})(Styles || (Styles = {}));
export var Colors;
(function (Colors) {
    Colors["CPU"] = "#0AA015";
    Colors["RAM"] = "#07630D";
    Colors["DISK"] = "#730F92";
    Colors["GPU"] = "#0C86F4";
    Colors["VRAM"] = "#176EC7";
    Colors["TEMP_START"] = "#00ff00";
    Colors["TEMP_END"] = "#ff0000";
    // TPU specific colors
    Colors["HBM"] = "#176EC7";
    Colors["DUTY_CYCLE"] = "#FF6B35";
    Colors["TENSORCORE"] = "#F7931E";
})(Colors || (Colors = {}));
