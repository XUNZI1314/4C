from dataclasses import dataclass


@dataclass(frozen=True)
class AppSettings:
    page_title: str = "蛋白质可视化原型软件"
    layout: str = "wide"
    viewer_width: int = 900
    viewer_height: int = 700
    default_opacity: float = 0.35
    neutral_color: str = "#c7c7c7"
    highlight_color: str = "yellow"
    background_color: str = "white"


SETTINGS = AppSettings()
