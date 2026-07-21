# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from generative_recommenders.dlrm_v3.configs import get_hstu_configs
from generative_recommenders.dlrm_v3.datasets.dataset import DLRMv3RandomDataset


class DatasetBatchingTest(unittest.TestCase):
    def test_preserves_multicharacter_contextual_feature_names(self) -> None:
        dataset = DLRMv3RandomDataset(get_hstu_configs("debug"))

        self.assertEqual(dataset.contexual_features, ["viewer_id", "dummy_contexual"])


if __name__ == "__main__":
    unittest.main()
