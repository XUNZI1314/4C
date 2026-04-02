from datetime import datetime


def build_analysis_summary(energy_table) -> dict:
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "residue_count": int(len(energy_table)),
        "min_energy": float(energy_table["delta_total"].min()),
        "max_energy": float(energy_table["delta_total"].max()),
        "mean_energy": float(energy_table["delta_total"].mean()),
        "lowest_residue": energy_table.loc[energy_table["delta_total"].idxmin(), "label"],
        "highest_residue": energy_table.loc[energy_table["delta_total"].idxmax(), "label"],
    }


def build_text_report(energy_table) -> str:
    summary = build_analysis_summary(energy_table)
    lines = [
        "蛋白质可视化分析报告",
        "=" * 30,
        f"生成时间: {summary['generated_at']}",
        f"残基总数: {summary['residue_count']}",
        f"最小能量: {summary['min_energy']:.3f}",
        f"最大能量: {summary['max_energy']:.3f}",
        f"平均能量: {summary['mean_energy']:.3f}",
        f"最低能量残基: {summary['lowest_residue']}",
        f"最高能量残基: {summary['highest_residue']}",
        "",
        "残基能量明细:",
    ]

    for row in energy_table.itertuples(index=False):
        lines.append(
            f"- {row.label}: DELTA TOTAL = {float(row.delta_total):.3f}, color = {row.heat_color}"
        )

    return "\n".join(lines)
