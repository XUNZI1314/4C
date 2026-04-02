import py3Dmol

from protein_visualizer.config.settings import SETTINGS


def build_view(
    pdb_text: str,
    energy_table,
    threshold: float,
    display_mode: str,
    show_backbone: bool,
    show_sidechain: bool,
    show_surface: bool,
    opacity: float,
    selected_chain: str,
    selected_resid: int,
):
    viewer = py3Dmol.view(width=SETTINGS.viewer_width, height=SETTINGS.viewer_height)
    viewer.addModel(pdb_text, "pdb")
    viewer.setBackgroundColor(SETTINGS.background_color)

    for row in energy_table.itertuples(index=False):
        color = row.heat_color if abs(float(row.delta_total)) >= threshold else SETTINGS.neutral_color
        residue_sel = {"chain": row.chain, "resi": int(row.resid)}

        if show_backbone:
            viewer.setStyle(residue_sel, {"cartoon": {"color": color, "opacity": 0.95}})
        if show_sidechain:
            viewer.addStyle(
                {"chain": row.chain, "resi": int(row.resid), "not": {"atom": ["N", "CA", "C", "O", "OXT"]}},
                {"stick": {"color": color, "radius": 0.18, "opacity": max(0.15, opacity)}},
            )
        if display_mode == "ball_stick":
            viewer.addStyle(
                residue_sel,
                {
                    "stick": {"color": color, "radius": 0.22, "opacity": max(0.2, opacity)},
                    "sphere": {"color": color, "scale": 0.28, "opacity": max(0.2, opacity)},
                },
            )
        elif display_mode == "surface" and show_surface:
            viewer.addSurface(py3Dmol.VDW, {"opacity": 0.45, "color": color}, residue_sel)

    if display_mode == "transparent" and show_surface:
        viewer.addSurface(py3Dmol.VDW, {"opacity": min(0.18, opacity), "color": SETTINGS.background_color})
    elif show_surface and display_mode != "surface":
        viewer.addSurface(py3Dmol.VDW, {"opacity": min(0.12, opacity), "color": "lightgray"})

    viewer.addStyle(
        {"chain": selected_chain, "resi": int(selected_resid)},
        {
            "stick": {"color": SETTINGS.highlight_color, "radius": 0.32},
            "sphere": {"color": SETTINGS.highlight_color, "scale": 0.35},
        },
    )
    viewer.zoomTo({"chain": selected_chain, "resi": int(selected_resid)})
    viewer.spin(False)
    return viewer
