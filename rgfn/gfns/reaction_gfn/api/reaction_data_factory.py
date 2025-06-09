import json
from copy import copy
from pathlib import Path
from typing import Dict, List, Tuple

import gin
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles

from rgfn.gfns.reaction_gfn.api.data_structures import (
    AnchoredReaction,
    Molecule,
    Reaction,
)
from rgfn.gfns.reaction_gfn.api.utils import read_txt_file


@gin.configurable()
class ReactionDataFactory:
    def __init__(
        self,
        reaction_path: str | Path,
        fragment_path: str | Path,
        cost_path: str | Path = None,
        yield_path: str | Path = None,
        yield_value: float | None = None,
    ):
        if yield_value and yield_path:
            raise ValueError("Yield_value and yield_path are mutually exclusive")
        reactions = read_txt_file(reaction_path)
        self.reactions = [Reaction(r, idx) for idx, r in enumerate(reactions)]
        self.disconnections = [reaction.reversed() for reaction in self.reactions]

        self.anchored_reactions = []
        self.reaction_anchor_map: Dict[Tuple[Reaction, int], AnchoredReaction] = {}
        self.anchor_to_reaction_map: Dict[AnchoredReaction, Reaction] = {}
        for reaction in self.reactions:
            for i in range(len(reaction.left_side_patterns)):
                anchored_reaction = AnchoredReaction(
                    reaction=reaction.reaction,
                    idx=len(self.anchored_reactions),
                    anchor_pattern_idx=i,
                )
                self.reaction_anchor_map[(reaction, i)] = anchored_reaction
                self.anchored_reactions.append(anchored_reaction)
                self.anchor_to_reaction_map[anchored_reaction] = reaction
        self.anchored_disconnections = [reaction.reversed() for reaction in self.anchored_reactions]

        fragments_list = read_txt_file(fragment_path)
        fragments_list = sorted(list(set(MolToSmiles(MolFromSmiles(x)) for x in fragments_list)))
        self.fragments = [Molecule(f, idx=idx) for idx, f in enumerate(fragments_list)]

        if cost_path is not None:
            self.fragment_to_cost = json.load(open(cost_path))
            self.fragment_to_cost = {
                Molecule(k): float(v) for k, v in self.fragment_to_cost.items()
            }
        else:
            self.fragment_to_cost = {}

        if yield_path is not None:
            df = pd.read_csv(yield_path, index_col=0)
            reaction_to_yield = {row["Reaction"]: row["yield"] for _, row in df.iterrows()}
            self.reaction_to_yield = {
                Reaction(k, idx=0): float(v) for k, v in reaction_to_yield.items()
            }
        else:
            self.reaction_to_yield = {}
        self.yield_value = yield_value

        print(
            f"Using {len(self.fragments)} fragments, {len(self.reactions)} reactions, and {len(self.anchored_reactions)} anchored reactions"
        )

    def get_reactions(self) -> List[Reaction]:
        return copy(self.reactions)

    def get_disconnections(self) -> List[Reaction]:
        return copy(self.disconnections)

    def get_anchored_reactions(self) -> List[AnchoredReaction]:
        return copy(self.anchored_reactions)

    def get_reaction_anchor_map(self) -> Dict[Tuple[Reaction, int], AnchoredReaction]:
        return copy(self.reaction_anchor_map)

    def get_anchor_to_reaction_map(self) -> Dict[AnchoredReaction, Reaction]:
        return copy(self.anchor_to_reaction_map)

    def get_anchored_disconnections(self) -> List[AnchoredReaction]:
        return copy(self.anchored_disconnections)

    def get_fragments(self) -> List[Molecule]:
        return copy(self.fragments)

    def get_fragment_to_cost(self) -> Dict[Molecule, float]:
        return copy(self.fragment_to_cost)

    def get_reaction_to_yield(self) -> Dict[Reaction, float]:
        return copy(self.reaction_to_yield)
