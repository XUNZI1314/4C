from dataclasses import dataclass


@dataclass(frozen=True)
class AppSettings:
    page_title: str = "蛋白质可视化原型软件"
    layout: str = "wide"
    viewer_width: int = 900
    viewer_height: int = 700
    # 增加默认不透明度以便在表面模式下更易观察
    default_opacity: float = 0.8
    neutral_color: str = "#c7c7c7"
    # 更醒目的高亮颜色
    highlight_color: str = "#ff5e57"
    background_color: str = "#f6f9fc"


SETTINGS = AppSettings()
