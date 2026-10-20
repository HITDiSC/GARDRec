from typing import List, Tuple

import numpy as np

from .model import GCNRecommendationDataset


class GCNRecommendationDatasetV2(GCNRecommendationDataset):
    """Dataset used by the v2 ranker.

    The base dataset left-pads prompts after building candidate spans.  V2 needs
    candidate-title token masks to be shifted by the same padding length so that
    masked pooling reads the real title tokens rather than padding positions.
    """

    def _candidate_text_mask(
        self,
        candidate_spans: List[Tuple[int, int]],
        prompt_token_length: int,
    ):
        text_mask = np.zeros(
            (self.num_candidates, self.max_prompt_length), dtype=np.float32
        )
        padding_length = max(
            0,
            self.max_prompt_length - min(prompt_token_length, self.max_prompt_length),
        )
        for index, (start, end) in enumerate(candidate_spans[: self.num_candidates]):
            clipped_start = min(start + padding_length, self.max_prompt_length)
            clipped_end = min(end + padding_length, self.max_prompt_length)
            if clipped_end > clipped_start:
                text_mask[index, clipped_start:clipped_end] = 1.0
        return text_mask
