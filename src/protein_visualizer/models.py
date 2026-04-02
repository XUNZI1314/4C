from dataclasses import dataclass


BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


@dataclass(frozen=True)
class ResidueKey:
    chain: str
    resid: int
    resname: str
