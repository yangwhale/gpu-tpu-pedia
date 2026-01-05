"""
@author: Crystian
@title: Crystools
@nickname: Crystools
@version: 1.27.4
@project: "https://github.com/crystian/comfyui-crystools",
@description: Plugins for multiples uses, mainly for debugging, you need them! IG: https://www.instagram.com/crystian.ia
"""

# ============================================================================
# 【Protobuf 冲突修复】2026-01-05
#
# 问题根因：
#   - JAX/tpu_info 初始化加载的 protobuf 版本与 SentencePiece (AutoTokenizer) 不兼容
#   - 两者同时存在会导致 SIGABRT 崩溃
#
# 解决方案：
#   - 在 JAX 导入之前先**实际加载** CogVideoX 的 tokenizer
#   - 仅导入 AutoTokenizer 类不够，必须实际加载模型的 tokenizer
#   - 让 SentencePiece 先初始化 protobuf，后续 JAX 会兼容这个版本
#
# 为什么在 Crystools 中添加：
#   - ComfyUI 按字母顺序加载 custom nodes
#   - "ComfyUI-Crystools" 排在 "ComfyUI-CogVideoX-TPU" 之前
#   - 因此在这里预加载可以确保 Tokenizer 在 JAX 之前初始化
#
# 注意：
#   - 必须在 `from .core import ...` 之前执行，因为 core 会初始化 JAX
#   - 需要实际加载 tokenizer（不只是导入类），才能触发 SentencePiece 初始化
# ============================================================================

print("[Crystools] Pre-loading Tokenizer for TPU protobuf conflict prevention...")
try:
    import transformers
    from transformers import AutoTokenizer
    
    # 【关键】实际加载 CogVideoX 的 tokenizer，触发 SentencePiece 完整初始化
    # 这会缓存 tokenizer，后续 CogVideoXPipeline 会复用缓存
    _preloaded_tokenizer = AutoTokenizer.from_pretrained(
        "zai-org/CogVideoX1.5-5B",
        subfolder="tokenizer"
    )
    print("[Crystools] ✓ CogVideoX tokenizer pre-loaded successfully (protobuf conflict prevention)")
    del _preloaded_tokenizer  # 释放内存，transformers 会自动缓存
except ImportError as e:
    print(f"[Crystools] Info: transformers not available ({e}), skipping tokenizer pre-load")
except Exception as e:
    # 如果 tokenizer 下载失败，尝试只导入类（至少触发部分初始化）
    print(f"[Crystools] Warning: Failed to pre-load CogVideoX tokenizer: {e}")
    print("[Crystools] Falling back to class-only import...")
    try:
        from transformers import AutoTokenizer, T5Tokenizer
        # 触发 T5Tokenizer 的模块初始化
        _ = T5Tokenizer
        print("[Crystools] ✓ Tokenizer classes imported (partial protobuf init)")
    except Exception as e2:
        print(f"[Crystools] Warning: Fallback also failed: {e2}")

from .core import version, logger
logger.info(f'Crystools version: {version}')

from .nodes._names import CLASSES
from .nodes.primitive import CBoolean, CText, CTextML, CInteger, CFloat
from .nodes.switch import CSwitchBooleanAny, CSwitchBooleanLatent, CSwitchBooleanConditioning, CSwitchBooleanImage, \
  CSwitchBooleanString, CSwitchBooleanMask, CSwitchFromAny
from .nodes.debugger import CConsoleAny, CConsoleAnyToJson
from .nodes.image import CImagePreviewFromImage, CImageLoadWithMetadata, CImageGetResolution, CImagePreviewFromMetadata, \
    CImageSaveWithExtraMetadata
from .nodes.list import CListAny, CListString
from .nodes.pipe import CPipeToAny, CPipeFromAny
from .nodes.utils import CUtilsCompareJsons, CUtilsStatSystem
from .nodes.metadata import CMetadataExtractor, CMetadataCompare
from .nodes.parameters import CJsonFile, CJsonExtractor
from .server import *
from .general import *

NODE_CLASS_MAPPINGS = {
    CLASSES.CBOOLEAN_NAME.value: CBoolean,
    CLASSES.CTEXT_NAME.value: CText,
    CLASSES.CTEXTML_NAME.value: CTextML,
    CLASSES.CINTEGER_NAME.value: CInteger,
    CLASSES.CFLOAT_NAME.value: CFloat,

    CLASSES.CDEBUGGER_CONSOLE_ANY_NAME.value: CConsoleAny,
    CLASSES.CDEBUGGER_CONSOLE_ANY_TO_JSON_NAME.value: CConsoleAnyToJson,

    CLASSES.CLIST_ANY_NAME.value: CListAny,
    CLASSES.CLIST_STRING_NAME.value: CListString,

    CLASSES.CSWITCH_FROM_ANY_NAME.value: CSwitchFromAny,
    CLASSES.CSWITCH_ANY_NAME.value: CSwitchBooleanAny,
    CLASSES.CSWITCH_LATENT_NAME.value: CSwitchBooleanLatent,
    CLASSES.CSWITCH_CONDITIONING_NAME.value: CSwitchBooleanConditioning,
    CLASSES.CSWITCH_IMAGE_NAME.value: CSwitchBooleanImage,
    CLASSES.CSWITCH_MASK_NAME.value: CSwitchBooleanMask,
    CLASSES.CSWITCH_STRING_NAME.value: CSwitchBooleanString,

    CLASSES.CPIPE_TO_ANY_NAME.value: CPipeToAny,
    CLASSES.CPIPE_FROM_ANY_NAME.value: CPipeFromAny,

    CLASSES.CIMAGE_LOAD_METADATA_NAME.value: CImageLoadWithMetadata,
    CLASSES.CIMAGE_GET_RESOLUTION_NAME.value: CImageGetResolution,
    CLASSES.CIMAGE_PREVIEW_IMAGE_NAME.value: CImagePreviewFromImage,
    CLASSES.CIMAGE_PREVIEW_METADATA_NAME.value: CImagePreviewFromMetadata,
    CLASSES.CIMAGE_SAVE_METADATA_NAME.value: CImageSaveWithExtraMetadata,

    CLASSES.CMETADATA_EXTRACTOR_NAME.value: CMetadataExtractor,
    CLASSES.CMETADATA_COMPARATOR_NAME.value: CMetadataCompare,
    CLASSES.CUTILS_JSON_COMPARATOR_NAME.value: CUtilsCompareJsons,
    CLASSES.CUTILS_STAT_SYSTEM_NAME.value: CUtilsStatSystem,
    CLASSES.CJSONFILE_NAME.value: CJsonFile,
    CLASSES.CJSONEXTRACTOR_NAME.value: CJsonExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    CLASSES.CBOOLEAN_NAME.value: CLASSES.CBOOLEAN_DESC.value,
    CLASSES.CTEXT_NAME.value: CLASSES.CTEXT_DESC.value,
    CLASSES.CTEXTML_NAME.value: CLASSES.CTEXTML_DESC.value,
    CLASSES.CINTEGER_NAME.value: CLASSES.CINTEGER_DESC.value,
    CLASSES.CFLOAT_NAME.value: CLASSES.CFLOAT_DESC.value,

    CLASSES.CDEBUGGER_CONSOLE_ANY_NAME.value: CLASSES.CDEBUGGER_ANY_DESC.value,
    CLASSES.CDEBUGGER_CONSOLE_ANY_TO_JSON_NAME.value: CLASSES.CDEBUGGER_CONSOLE_ANY_TO_JSON_DESC.value,

    CLASSES.CLIST_ANY_NAME.value: CLASSES.CLIST_ANY_DESC.value,
    CLASSES.CLIST_STRING_NAME.value: CLASSES.CLIST_STRING_DESC.value,

    CLASSES.CSWITCH_FROM_ANY_NAME.value: CLASSES.CSWITCH_FROM_ANY_DESC.value,
    CLASSES.CSWITCH_ANY_NAME.value: CLASSES.CSWITCH_ANY_DESC.value,
    CLASSES.CSWITCH_LATENT_NAME.value: CLASSES.CSWITCH_LATENT_DESC.value,
    CLASSES.CSWITCH_CONDITIONING_NAME.value: CLASSES.CSWITCH_CONDITIONING_DESC.value,
    CLASSES.CSWITCH_IMAGE_NAME.value: CLASSES.CSWITCH_IMAGE_DESC.value,
    CLASSES.CSWITCH_MASK_NAME.value: CLASSES.CSWITCH_MASK_DESC.value,
    CLASSES.CSWITCH_STRING_NAME.value: CLASSES.CSWITCH_STRING_DESC.value,

    CLASSES.CPIPE_TO_ANY_NAME.value: CLASSES.CPIPE_TO_ANY_DESC.value,
    CLASSES.CPIPE_FROM_ANY_NAME.value: CLASSES.CPIPE_FROM_ANY_DESC.value,

    CLASSES.CIMAGE_LOAD_METADATA_NAME.value: CLASSES.CIMAGE_LOAD_METADATA_DESC.value,
    CLASSES.CIMAGE_GET_RESOLUTION_NAME.value: CLASSES.CIMAGE_GET_RESOLUTION_DESC.value,
    CLASSES.CIMAGE_PREVIEW_IMAGE_NAME.value: CLASSES.CIMAGE_PREVIEW_IMAGE_DESC.value,
    CLASSES.CIMAGE_PREVIEW_METADATA_NAME.value: CLASSES.CIMAGE_PREVIEW_METADATA_DESC.value,
    CLASSES.CIMAGE_SAVE_METADATA_NAME.value: CLASSES.CIMAGE_SAVE_METADATA_DESC.value,

    CLASSES.CMETADATA_EXTRACTOR_NAME.value: CLASSES.CMETADATA_EXTRACTOR_DESC.value,
    CLASSES.CMETADATA_COMPARATOR_NAME.value: CLASSES.CMETADATA_COMPARATOR_DESC.value,

    CLASSES.CUTILS_JSON_COMPARATOR_NAME.value: CLASSES.CUTILS_JSON_COMPARATOR_DESC.value,
    CLASSES.CUTILS_STAT_SYSTEM_NAME.value: CLASSES.CUTILS_STAT_SYSTEM_DESC.value,

    CLASSES.CJSONFILE_NAME.value: CLASSES.CJSONFILE_DESC.value,
    CLASSES.CJSONEXTRACTOR_NAME.value: CLASSES.CJSONEXTRACTOR_DESC.value,
}


WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
