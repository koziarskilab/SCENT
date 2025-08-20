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
        last_step_reaction_path: str | Path = None,
        last_step_fragment_path: str | Path = None,
        cost_path: str | Path = None,
        yield_path: str | Path = None,
        yield_value: float | None = None,
    ):
        if yield_value and yield_path:
            raise ValueError("Yield_value and yield_path are mutually exclusive")
        reactions = read_txt_file(reaction_path)

        # Handle last step reactions (must be subset of reactions)
        if last_step_reaction_path:
            lsd_reactions = read_txt_file(last_step_reaction_path)
            assert all(
                r in reactions for r in lsd_reactions
            ), "Last step reactions must be a subset of the reactions"
            div_reaction_idxs = [
                idx for idx in range(len(reactions)) if reactions[idx] in lsd_reactions
            ]
        else:
            div_reaction_idxs = list(range(len(reactions)))

        self.div_reaction_idxs = div_reaction_idxs

        self.reactions = [Reaction(r, idx) for idx, r in enumerate(reactions)]
        self.disconnections = [reaction.reversed() for reaction in self.reactions]

        self.anchored_reactions = []
        self.anchored_div_reaction_idxs = []
        self.reaction_anchor_map: Dict[Tuple[Reaction, int], AnchoredReaction] = {}
        self.anchor_to_reaction_map: Dict[AnchoredReaction, Reaction] = {}
        for idx, reaction in enumerate(self.reactions):
            for i in range(len(reaction.left_side_patterns)):
                anchored_reaction = AnchoredReaction(
                    reaction=reaction.reaction,
                    idx=len(self.anchored_reactions),
                    anchor_pattern_idx=i,
                )
                if idx in div_reaction_idxs:
                    self.anchored_div_reaction_idxs.append(len(self.anchored_reactions))
                self.reaction_anchor_map[(reaction, i)] = anchored_reaction
                self.anchored_reactions.append(anchored_reaction)
                self.anchor_to_reaction_map[anchored_reaction] = reaction
        self.anchored_disconnections = [reaction.reversed() for reaction in self.anchored_reactions]

        fragments_list = read_txt_file(fragment_path)
        fragments_list = sorted(list(set(MolToSmiles(MolFromSmiles(x)) for x in fragments_list)))
        self.fragments = [Molecule(f, idx=idx) for idx, f in enumerate(fragments_list)]

        if last_step_fragment_path:
            lsd_fragments_list = read_txt_file(last_step_fragment_path)
            lsd_fragments_list = sorted(
                list(set(MolToSmiles(MolFromSmiles(x)) for x in lsd_fragments_list))
            )
            lsd_fragments_list = [f for f in lsd_fragments_list if f not in fragments_list]
            self.div_fragments_idxs = [
                idx + len(self.fragments) for idx in range(len(lsd_fragments_list))
            ]
            lsd_fragments_list = [
                Molecule(f, idx=idx) for idx, f in zip(self.div_fragments_idxs, lsd_fragments_list)
            ]
            self.fragments.extend(lsd_fragments_list)
        else:
            self.div_fragments_idxs = []

        assert all(
            f.num_reactions == 0 for f in self.fragments
        ), "Basic fragments should require only one reaction"

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
            f"Using {len(self.fragments) - len(self.div_fragments_idxs)} fragments, {len(self.reactions)} reactions, and {len(self.anchored_reactions)} anchored reactions"
        )
        print(
            f"Using {len(self.div_fragments_idxs)} div fragments, {len(self.div_reaction_idxs)} div reactions, and {len(self.anchored_div_reaction_idxs)} anchored div reactions"
        )

    def get_reactions(self) -> List[Reaction]:
        return copy(self.reactions)

    def get_div_reaction_idxs(self) -> List[int]:
        return copy(self.div_reaction_idxs)

    def get_disconnections(self) -> List[Reaction]:
        return copy(self.disconnections)

    def get_anchored_reactions(self) -> List[AnchoredReaction]:
        return copy(self.anchored_reactions)

    def get_anchored_div_reaction_idxs(self) -> List[int]:
        return copy(self.anchored_div_reaction_idxs)

    def get_reaction_anchor_map(self) -> Dict[Tuple[Reaction, int], AnchoredReaction]:
        return copy(self.reaction_anchor_map)

    def get_anchor_to_reaction_map(self) -> Dict[AnchoredReaction, Reaction]:
        return copy(self.anchor_to_reaction_map)

    def get_anchored_disconnections(self) -> List[AnchoredReaction]:
        return copy(self.anchored_disconnections)

    def get_fragments(self) -> List[Molecule]:
        return copy(self.fragments)
    
    def get_div_fragments_idxs(self) -> List[Molecule]:
        return copy(self.div_fragments_idxs)

    def get_fragment_to_cost(self) -> Dict[Molecule, float]:
        return copy(self.fragment_to_cost)

    def get_reaction_to_yield(self) -> Dict[Reaction, float]:
        return copy(self.reaction_to_yield)
