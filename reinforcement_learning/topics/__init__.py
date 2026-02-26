"""Auto-discovers all topic modules in the reinforcement_learning package."""
import importlib, pkgutil
from pathlib import Path

def get_all_topics() -> dict:
    topics = {}
    package_dir = Path(__file__).parent
    for finder, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
        if module_name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"reinforcement_learning.topics.{module_name}")
            display = getattr(mod, "DISPLAY_NAME", module_name.replace("_", " ").title())
            topics[display] = {
                "icon":        getattr(mod, "ICON",        "📖"),
                "subtitle":    getattr(mod, "SUBTITLE",    ""),
                "theory":      getattr(mod, "THEORY",      "_Coming soon._"),
                "operations":  getattr(mod, "OPERATIONS",  {}),
                "visual_html": getattr(mod, "VISUAL_HTML", ""),
            }
        except Exception as e:
            print(f"[WARN] Could not load {module_name}: {e}")
    return topics
