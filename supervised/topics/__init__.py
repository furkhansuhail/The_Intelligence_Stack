"""Auto-discovers all topic modules in this package."""

import importlib.util
from pathlib import Path


def get_all_topics() -> dict:
    topics = {}
    package_dir = Path(__file__).parent

    for py_file in sorted(package_dir.glob("*.py")):
        # Skip __init__.py and any private/utility files
        if py_file.name.startswith("_"):
            continue

        module_name = py_file.stem  # e.g. "01_linear_regression"

        try:
            # ── Load by file path so digit-prefixed names (e.g. 01_...) work ──
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # ── Try get_content() first ───────────────────────────────────────
            data = None
            if callable(getattr(mod, "get_content", None)):
                try:
                    data = mod.get_content()
                except Exception as e:
                    print(f"[WARN] get_content() failed for '{module_name}': {e}")
                    data = None  # fall through to legacy path below

            # ── Use get_content() result if we got one ────────────────────────
            if data is not None:
                display = data.get("display_name",
                                   module_name.replace("_", " ").title())
                topics[display] = {
                    "icon":          data.get("icon",          "📖"),
                    "subtitle":      data.get("subtitle",       ""),
                    "theory":        data.get("theory",         "_Coming soon._"),
                    "operations":    data.get("operations",     {}),
                    "visual_html":   data.get("visual_html",    ""),
                    "visual_height": data.get("visual_height",  700),
                    "complexity":    data.get("complexity",     ""),
                }

            # ── Legacy fallback: read module-level attributes ─────────────────
            else:
                display = getattr(mod, "DISPLAY_NAME",
                                  module_name.replace("_", " ").title())
                topics[display] = {
                    "icon":          getattr(mod, "ICON",          "📖"),
                    "subtitle":      getattr(mod, "SUBTITLE",      ""),
                    "theory":        getattr(mod, "THEORY",        "_Coming soon._"),
                    "operations":    getattr(mod, "OPERATIONS",    {}),
                    "visual_html":   getattr(mod, "VISUAL_HTML",   ""),
                    "visual_height": getattr(mod, "VISUAL_HEIGHT", 700),
                    "complexity":    getattr(mod, "COMPLEXITY",    ""),
                }

        except Exception as e:
            print(f"[WARN] Could not load supervised topic '{module_name}': {e}")

    return topics

# """Auto-discovers all topic modules in this package."""
#
# import importlib
# import pkgutil
# from pathlib import Path
#
#
# def get_all_topics() -> dict:
#     topics = {}
#     package_dir = Path(__file__).parent
#     for finder, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
#         if module_name.startswith("_"):
#             continue
#         try:
#             mod = importlib.import_module(f"supervised.topics.{module_name}")
#
#             # ── Try get_content() first ───────────────────────────────────────
#             data = None
#             if callable(getattr(mod, "get_content", None)):
#                 try:
#                     data = mod.get_content()
#                 except Exception as e:
#                     print(f"[WARN] get_content() failed for '{module_name}': {e}")
#                     data = None  # fall through to legacy path below
#
#             # ── Use get_content() result if we got one ────────────────────────
#             if data is not None:
#                 display = data.get("display_name",
#                                    module_name.replace("_", " ").title())
#                 topics[display] = {
#                     "icon":          data.get("icon",          "📖"),
#                     "subtitle":      data.get("subtitle",       ""),
#                     "theory":        data.get("theory",         "_Coming soon._"),
#                     "operations":    data.get("operations",     {}),
#                     "visual_html":   data.get("visual_html",    ""),
#                     "visual_height": data.get("visual_height",  700),
#                     "complexity":    data.get("complexity",     ""),
#                 }
#
#             # ── Legacy fallback: read module-level attributes ─────────────────
#             else:
#                 display = getattr(mod, "DISPLAY_NAME",
#                                   module_name.replace("_", " ").title())
#                 topics[display] = {
#                     "icon":          getattr(mod, "ICON",          "📖"),
#                     "subtitle":      getattr(mod, "SUBTITLE",      ""),
#                     "theory":        getattr(mod, "THEORY",        "_Coming soon._"),
#                     "operations":    getattr(mod, "OPERATIONS",    {}),
#                     "visual_html":   getattr(mod, "VISUAL_HTML",   ""),
#                     "visual_height": getattr(mod, "VISUAL_HEIGHT", 700),
#                     "complexity":    getattr(mod, "COMPLEXITY",    ""),
#                 }
#
#         except Exception as e:
#             print(f"[WARN] Could not load supervised topic '{module_name}': {e}")
#     return topics
#
# # """Auto-discovers all topic modules in this package."""
# #
# # import importlib
# # import pkgutil
# # from pathlib import Path
# #
# #
# # def get_all_topics() -> dict:
# #     topics = {}
# #     package_dir = Path(__file__).parent
# #     for finder, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
# #         if module_name.startswith("_"):
# #             continue
# #         try:
# #             mod = importlib.import_module(f"supervised.topics.{module_name}")
# #
# #             # ── Prefer get_content() if the module defines it ─────────────────
# #             # Newer modules return everything via get_content(). Older modules
# #             # expose module-level attributes directly. Support both.
# #             if callable(getattr(mod, "get_content", None)):
# #                 data = mod.get_content()
# #                 topics[data["display_name"]] = {
# #                     "icon":          data.get("icon",          "📖"),
# #                     "subtitle":      data.get("subtitle",       ""),
# #                     "theory":        data.get("theory",         "_Coming soon._"),
# #                     "operations":    data.get("operations",     {}),
# #                     "visual_html":   data.get("visual_html",    ""),
# #                     "visual_height": data.get("visual_height",  700),
# #                     "complexity":    data.get("complexity",     ""),
# #                 }
# #             else:
# #                 # ── Legacy fallback: read module-level attributes ──────────────
# #                 display = getattr(mod, "DISPLAY_NAME", module_name.replace("_", " ").title())
# #                 topics[display] = {
# #                     "icon":          getattr(mod, "ICON",          "📖"),
# #                     "subtitle":      getattr(mod, "SUBTITLE",      ""),
# #                     "theory":        getattr(mod, "THEORY",        "_Coming soon._"),
# #                     "operations":    getattr(mod, "OPERATIONS",    {}),
# #                     "visual_html":   getattr(mod, "VISUAL_HTML",   ""),
# #                     "visual_height": getattr(mod, "VISUAL_HEIGHT", 700),
# #                     "complexity":    getattr(mod, "COMPLEXITY",    ""),
# #                 }
# #
# #         except Exception as e:
# #             print(f"[WARN] Could not load supervised topic '{module_name}': {e}")
# #     return topics
# #
# # # """Auto-discovers all topic modules in this package."""
# # #
# # # import importlib
# # # import pkgutil
# # # from pathlib import Path
# # #
# # #
# # # def get_all_topics() -> dict:
# # #     topics = {}
# # #     package_dir = Path(__file__).parent
# # #     for finder, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
# # #         if module_name.startswith("_"):
# # #             continue
# # #         try:
# # #             mod = importlib.import_module(f"supervised.topics.{module_name}")
# # #             display = getattr(mod, "DISPLAY_NAME", module_name.replace("_", " ").title())
# # #             topics[display] = {
# # #                 "icon":       getattr(mod, "ICON",     "📖"),
# # #                 "subtitle":   getattr(mod, "SUBTITLE", ""),
# # #                 "theory":     getattr(mod, "THEORY",   "_Coming soon._"),
# # #                 "operations": getattr(mod, "OPERATIONS", {}),
# # #                 "visual_html":getattr(mod, "VISUAL_HTML", ""),
# # #             }
# # #         except Exception as e:
# # #             print(f"[WARN] Could not load supervised topic '{module_name}': {e}")
# # #     return topics
